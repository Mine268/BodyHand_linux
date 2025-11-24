#include <exception>
#include <algorithm>
#include <opencv2/sfm/triangulation.hpp>
#include "Pose3D.h"

namespace BodyHand {

	PoseEstimator::PoseEstimator(
		BodyModelConfig _body_model_cfg,
		HandModelConfig _hand_model_cfg,
		int _num_views,
		const std::vector<cv::Mat>& _intrinsics,
		const std::vector<cv::Mat>& _rot_transformations,
		const std::vector<cv::Mat>& _transl_transformation,
		const std::vector<std::vector<float>>& _undists
	) :
		body_model_cfg(_body_model_cfg),
		hand_model_cfg(_hand_model_cfg),
		num_views(_num_views)
	{
		// 参数检查
		if (!(
			this->num_views == _intrinsics.size() &&
			this->num_views == _rot_transformations.size() &&
			this->num_views == _transl_transformation.size() &&
			this->num_views == _undists.size()
			)) {
			throw std::runtime_error("Camera count doesn't match");
		}
		if (std::any_of(_undists.begin(), _undists.end(),
			[](auto &&innerVec) {
				return innerVec.size() != 5;
			})) {
			throw std::runtime_error("Expect #undistortion coef to be 5");
		}

		// 加载相机内参
		for (std::size_t i = 0; i < _intrinsics.size(); ++i) {
			this->cameras.emplace_back(
				_intrinsics[i],
				_undists[i],
				_rot_transformations[i],
				_transl_transformation[i]
			);
		}

		// 加载模型
		if (!loadBodyModel()) {
			throw std::runtime_error("Cannot load body model");
		}
		if (!loadHandModel()) {
			throw std::runtime_error("Cannot load hand model");
		}
	}

	bool PoseEstimator::loadBodyModel() {
		return body_model.ReadModel(body_model_cfg.model_path, true, 0, true);
	}

	bool PoseEstimator::loadHandModel() {
		return hand_model.loadModel(hand_model_cfg.handlr_path, hand_model_cfg.hamer_path);
	}

	bool PoseEstimator::estimateBody(
		IN std::vector<cv::Mat>& imgs,
		OUT std::vector<std::vector<std::vector<cv::Point2f>>>& kpss2d,
		OUT std::vector<std::vector<std::vector<float>>>& conf_kpss,
		OUT std::vector<std::vector<float>>& conf_bodies
	) {
		kpss2d.clear();
		conf_kpss.clear();
		conf_bodies.clear();
		for (auto &img : imgs) {
			std::vector<OutputPose> poses_2d;
			if (!body_model.OnnxDetect(img, poses_2d)) {
				return false;
			}
			std::vector<std::vector<cv::Point2f>> view2d; // 每个图像中的所有二维姿态
			std::vector<std::vector<float>> conf_kps;
			std::vector<float> conf_body;
			for (auto &pose_2d : poses_2d) {
				std::vector<cv::Point2f> man2d; // 单个人的二维姿态
				std::vector<float> man_conf; // 单个人的关节点置信度
				for (std::size_t j = 0; j < pose_2d.kps.size(); j += 3) {
					man2d.emplace_back(pose_2d.kps[j], pose_2d.kps[j + 1]);
					man_conf.emplace_back(pose_2d.kps[j + 2]);
				}
				view2d.emplace_back(std::move(man2d));
				conf_kps.emplace_back(std::move(man_conf));
				conf_body.emplace_back(pose_2d.confidence);
			}
			kpss2d.emplace_back(std::move(view2d));
			conf_kpss.emplace_back(std::move(conf_kps));
			conf_bodies.emplace_back(std::move(conf_body));
		}
		// kpss2d，conf_kpss，conf_bodies的每个vector元素按照conf_bodies的值进行从大到小排序
		for (std::size_t i = 0; i < conf_bodies.size(); ++i) {
			// 获取当前图像的姿态数量 N
			std::size_t num_poses = conf_bodies[i].size();

			// 如果该图像没有人，则跳过
			if (num_poses == 0) {
				continue;
			}

			// 1. 创建一个索引向量，用于存储原始顺序
			std::vector<std::size_t> indices(num_poses);
			std::iota(indices.begin(), indices.end(), 0); // 填充 0, 1, 2, ..., N-1

			// 2. 使用 std::sort 和自定义比较函数对索引进行排序
			// 比较函数基于 conf_bodies[i] 的值进行降序 (从大到小)
			std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
				return conf_bodies[i][a] > conf_bodies[i][b];
				});

			// 3. 根据排序后的索引创建新的向量
			std::vector<std::vector<cv::Point2f>> sorted_kpss2d_view;
			std::vector<std::vector<float>> sorted_conf_kpss_view;
			std::vector<float> sorted_conf_bodies_view;

			for (std::size_t index : indices) {
				sorted_kpss2d_view.emplace_back(std::move(kpss2d[i][index]));
				sorted_conf_kpss_view.emplace_back(std::move(conf_kpss[i][index]));
				sorted_conf_bodies_view.emplace_back(conf_bodies[i][index]); // float可以直接复制
			}

			// 4. 用新的排序结果替换原始向量
			kpss2d[i] = std::move(sorted_kpss2d_view);
			conf_kpss[i] = std::move(sorted_conf_kpss_view);
			conf_bodies[i] = std::move(sorted_conf_bodies_view);
		}

		return true;
	}

	std::tuple<bool, bool> PoseEstimator::estimateHand(
		IN cv::Mat& img,
		OUT std::vector<cv::Point3f>& _kps_cam,
		OUT std::vector<cv::Point2f>& _kps_img,
		OUT std::optional<std::reference_wrapper<std::vector<cv::Rect2f>>> hand_bbox,
		IN int view_ix
	) {
		if (view_ix >= cameras.size()) {
			throw std::runtime_error("view_ix out of range");
		}

		auto [valid_left, valid_right] = this->hand_model.detectPose(
			img,
			cameras[view_ix].intrinsics,
			cameras[view_ix].undist,
			_kps_cam,
			_kps_img,
			hand_bbox
		);

		return { valid_left, valid_right };
	}

	std::vector<cv::Point3f> PoseEstimator::triangulate2DPoints(const std::vector<std::vector<cv::Point2f>>& img_coords) {
		// 构造投影矩阵
		std::vector<cv::Mat_<double>> projections_double;
		for (const auto& cam : cameras) {
			cv::Mat proj, proj_double;
			cv::hconcat(cam.rot_transformation, cam.transl_transformation, proj);
			proj = cam.intrinsics * proj;
			proj.convertTo(proj_double, CV_64F);
			projections_double.emplace_back(std::move(proj_double));
		}
		std::vector<cv::Mat_<double>> img_coords_double;
		img_coords_double.reserve(img_coords.size());
		for (const auto& coords : img_coords) {
			const int N = coords.size();
			cv::Mat_<double> mat(2, N);
			for (int i = 0; i < N; ++i) {
				mat.at<double>(0, i) = coords[i].x;
				mat.at<double>(1, i) = coords[i].y;
			}
			img_coords_double.emplace_back(std::move(mat));
		}
		cv::Mat kps_3d_double;
		cv::sfm::triangulatePoints(img_coords_double, projections_double, kps_3d_double);
		std::vector<cv::Point3f> kps_3d(kps_3d_double.cols);
		for (int i = 0; i < kps_3d_double.cols; ++i) {
			kps_3d[i] = cv::Point3f{
				static_cast<float>(kps_3d_double.at<double>(0, i)),
				static_cast<float>(kps_3d_double.at<double>(1, i)),
				static_cast<float>(kps_3d_double.at<double>(2, i))
			};
		}
		return kps_3d;
	}

	std::tuple<bool, bool, bool> PoseEstimator::estimatePose(
		IN std::vector<cv::Mat>& imgs,
		OUT std::vector<cv::Point3f>& body_kps,
		OUT std::vector<cv::Point3f>& hand_kps,
		IN int hand_ref_view
	) {
		if (hand_ref_view >= cameras.size()) {
			throw std::runtime_error("view_ix out of range");
		}

		hand_kps.clear();
		hand_kps.resize(42);

		// 人体姿态估计
		std::vector<std::vector<std::vector<cv::Point2f>>> body_kps_2d;
		std::vector<std::vector<std::vector<float>>> body_kps_conf;
		std::vector<std::vector<float>> body_conf;
		bool valid_body = estimateBody(imgs, body_kps_2d, body_kps_conf, body_conf);

		// 每个视图默认选第0个人
		std::vector<std::vector<cv::Point2f>> body_kps_2d_selected;
		for (const auto& k2d : body_kps_2d) {
			body_kps_2d_selected.emplace_back(k2d[0]);
		}

		// 多视图三角化
		std::vector<cv::Point3f> body_kps_3d;
		body_kps_3d = triangulate2DPoints(body_kps_2d_selected);

		// 手部姿态估计
		std::vector<cv::Point3f> hand_kps_3d;
		std::vector<cv::Point2f> hand_kps_2d;
		auto [valid_left, valid_right] = estimateHand(imgs[hand_ref_view], hand_kps_3d, hand_kps_2d, std::nullopt, hand_ref_view);

		body_kps = body_kps_3d;
		// 拼接
		if (valid_left) {
			std::transform(hand_kps_3d.begin(), hand_kps_3d.begin() + 21, hand_kps.begin(),
				[&](const cv::Point3f& joint) {
					// 左手拼接到左手手腕
					return joint - hand_kps_3d[0] + body_kps[9];
				}
			);
		}
		if (valid_right) {
			std::transform(hand_kps_3d.begin() + 21, hand_kps_3d.end(), hand_kps.begin() + 21,
				[&](const cv::Point3f& joint) {
					// 右手拼接到右手手腕
					return joint - hand_kps_3d[21] + body_kps[10];
				}
			);
		}

		return { valid_body, valid_left, valid_right };
	}

	void PoseEstimator::estimatePose(
		IN std::vector<cv::Mat>& imgs,
		OUT PoseResult& pose_result,
		IN int hand_ref_view
	) {
		if (hand_ref_view >= cameras.size()) {
			throw std::runtime_error("view_ix out of range");
		}

		// 人体姿态估计
		{
			std::vector<std::vector<std::vector<cv::Point2f>>> body_kps_2d;
			std::vector<std::vector<std::vector<float>>> body_kps_conf;
			std::vector<std::vector<float>> body_conf;
			bool valid_body = estimateBody(imgs, body_kps_2d, body_kps_conf, body_conf); // 返回值表示推理是否完成
			bool all_detect_2d = std::all_of(body_kps_2d.begin(), body_kps_2d.end(),
				[](const auto& kps) { return !kps.empty(); }); // 所有视图都检测到人体

			// 每个视图默认选第0个人
			std::vector<cv::Point3f> body_kps_3d;
			if (all_detect_2d) {
				std::vector<std::vector<cv::Point2f>> body_kps_2d_selected;
				for (const auto& k2d : body_kps_2d) {
					body_kps_2d_selected.emplace_back(k2d[0]);
				}
				// 多视图三角化
				body_kps_3d = triangulate2DPoints(body_kps_2d_selected);
			}

			// 结果存入
			pose_result.valid_body = valid_body && all_detect_2d; // 表示三维姿态是否有效
			pose_result.body_kps_3d = std::move(body_kps_3d);
			pose_result.body_kps_2d = std::move(body_kps_2d);
			pose_result.body_kps_conf = std::move(body_kps_conf);
			pose_result.body_conf = std::move(body_conf);
		}

		// 人手姿态估计
		{
			std::vector<cv::Point3f> hand_kps_3d;
			std::vector<cv::Point2f> hand_kps_2d;
			std::vector<cv::Rect2f> hand_bbox;
			auto [valid_left, valid_right] = estimateHand(
				imgs[hand_ref_view],
				hand_kps_3d,
				hand_kps_2d,
				hand_bbox,
				hand_ref_view
			);
			pose_result.valid_left = valid_left;
			pose_result.valid_right = valid_right;
			pose_result.hand_kps_3d = std::move(hand_kps_3d);
			pose_result.hand_kps_2d = std::move(hand_kps_2d);
			pose_result.hand_bbox = std::move(hand_bbox);
		}

		// 拼接
		if (pose_result.valid_body && pose_result.valid_left && pose_result.valid_right) {
			for (int i = 20; i >= 0; --i) {
				pose_result.hand_kps_3d[i] = pose_result.hand_kps_3d[i] - pose_result.hand_kps_3d[0] + pose_result.body_kps_3d[9];
				pose_result.hand_kps_3d[i + 21] = pose_result.hand_kps_3d[i + 21] - pose_result.hand_kps_3d[21] + pose_result.body_kps_3d[10];
			}
		}

		return;
	}

}