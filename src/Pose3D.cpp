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
		num_views(_num_views),
		yolo_det(body_model_cfg.yolo_path, cv::Size(640, 640), 0.35f, 0.45f),
		vit_pose(body_model_cfg.model_path, cv::Size(192, 256), "cuda")
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
		// return body_model.ReadModel(body_model_cfg.model_path, true, 0, true);
		// try {
		// 	yolo_det = YoloOnnx(body_model_cfg.yolo_path, cv::Size(640, 640), 0.35f, 0.45f); // conf_thres, iou_thres, allowed classes
		// 	vit_pose = VitInference(body_model_cfg.model_path, cv::Size(192, 256), "cuda");
		// }
		// catch (const std::exception& e) {
		// 	return false;
		// }
		return true;
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

		for (auto &img_bgr : imgs) {
			cv::Mat img_rgb;
			cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

			// ---------------------
			// 1. YOLO 人体检测
			// ---------------------
			auto dets = yolo_det.infer(img_bgr, {0});  // 只检测 person 类别
			if (dets.empty()) {
				kpss2d.emplace_back();
				conf_kpss.emplace_back();
				conf_bodies.emplace_back();
				continue;
			}

			// ---------------------
			// 2. ViTPose 关键点检测
			// ---------------------
			auto kpt_map = vit_pose.inference(img_rgb, dets);

			// ---------------------
			// 3. 转换为原 estimateBody 输出结构
			// ---------------------
			std::vector<std::vector<cv::Point2f>> view2d;
			std::vector<std::vector<float>> view_conf_kps;
			std::vector<float> view_conf_body;

			for (size_t i = 0; i < dets.size(); ++i) {
				if (kpt_map.find(i) == kpt_map.end()) continue;
				const auto &mat = kpt_map.at(i);  // 17x3

				std::vector<cv::Point2f> pts2d;
				std::vector<float> confs;

				for (int j = 0; j < mat.rows; ++j) {
					pts2d.emplace_back(mat.at<float>(j,0), mat.at<float>(j,1));
					confs.emplace_back(mat.at<float>(j,2));
				}

				view2d.emplace_back(std::move(pts2d));
				view_conf_kps.emplace_back(std::move(confs));
				view_conf_body.emplace_back(dets[i].score);
			}

			kpss2d.emplace_back(std::move(view2d));
			conf_kpss.emplace_back(std::move(view_conf_kps));
			conf_bodies.emplace_back(std::move(view_conf_body));
		}

		// ---------------------
		// 4. 对每个视图按人体置信度排序 (保持你的逻辑)
		// ---------------------
		for (std::size_t i = 0; i < conf_bodies.size(); ++i) {
			size_t N = conf_bodies[i].size();
			if (N == 0) continue;

			std::vector<size_t> idx(N);
			std::iota(idx.begin(), idx.end(), 0);

			std::sort(idx.begin(), idx.end(),
				[&](size_t a, size_t b) {
					return conf_bodies[i][a] > conf_bodies[i][b];
				});

			std::vector<std::vector<cv::Point2f>> sorted_kpss2d_view;
			std::vector<std::vector<float>>       sorted_conf_kpss_view;
			std::vector<float>                    sorted_conf_bodies_view;

			for (auto id : idx) {
				sorted_kpss2d_view.emplace_back(std::move(kpss2d[i][id]));
				sorted_conf_kpss_view.emplace_back(std::move(conf_kpss[i][id]));
				sorted_conf_bodies_view.emplace_back(conf_bodies[i][id]);
			}

			kpss2d[i]       = std::move(sorted_kpss2d_view);
			conf_kpss[i]    = std::move(sorted_conf_kpss_view);
			conf_bodies[i]  = std::move(sorted_conf_bodies_view);
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