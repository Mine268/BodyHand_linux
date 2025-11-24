#pragma once

#include <vector>
#include <string>
#include <optional>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "YoloONNX.h"
#include "HaMeRONNX.h"

#define IN
#define OUT

namespace BodyHand
{
	struct CameraParameter {
		cv::Mat intrinsics;
		std::vector<float> undist;
		cv::Mat rot_transformation;
		cv::Mat transl_transformation;
	};

	struct BodyModelConfig {
		std::string model_path;
	};

	struct HandModelConfig {
		std::string handlr_path;
		std::string hamer_path;
	};

	struct PoseResult {
		bool valid_body{ false }, valid_left{ false }, valid_right{ false };
		std::vector<cv::Point3f> body_kps_3d{};
		std::vector<std::vector<std::vector<cv::Point2f>>> body_kps_2d{};
		std::vector<std::vector<std::vector<float>>> body_kps_conf{};
		std::vector<std::vector<float>> body_conf{};
		std::vector<cv::Point3f> hand_kps_3d{};
		std::vector<cv::Point2f> hand_kps_2d{};
		std::vector<cv::Rect2f> hand_bbox{}; // xywh

		bool handValid() const {
			return valid_left && valid_right;
		}

		bool bodyValid() const {
			return valid_body;
		}

		bool allValid() const {
			return valid_body && (valid_left && valid_right);
		}
	};

	/// <summary>
	/// 手+身体关节点检测。需要标定的多视图。
	/// </summary>
	class PoseEstimator {
	public:
		PoseEstimator(
			BodyModelConfig _body_model_cfg,
			HandModelConfig _hand_model_cfg,
			int _num_views,
			const std::vector<cv::Mat>& _intrinsics,
			const std::vector<cv::Mat>& _rot_transformations,
			const std::vector<cv::Mat>& _transl_transformation,
			const std::vector<std::vector<float>>& _undists
		);
		~PoseEstimator() {}

		/// <summary>
		/// 估计图像中的二维人体姿态
		/// </summary>
		/// <param name="imgs">图像列表</param>
		/// <param name="kpss2d">每张图像中的所有人的人体姿态</param>
		/// <param name="conf_kpss">每个关节点的置信度</param>
		/// <param name="conf_bodies">每个人检测的置信度</param>
		/// <returns>返回结果是否有效</returns>
		bool estimateBody(
			IN std::vector<cv::Mat>& imgs,
			OUT std::vector<std::vector<std::vector<cv::Point2f>>>& kpss2d,
			OUT std::vector<std::vector<std::vector<float>>>& conf_kpss,
			OUT std::vector<std::vector<float>>& conf_bodies
		);

		/// <summary>
		/// 进行手部姿态估计，最多估计一个左手和右手
		/// </summary>
		/// <param name="img">单张待估计图像</param>
		/// <param name="_kps_cam">长度42，分别是左手+右手在相机空间中的位置</param>
		/// <param name="_kps_img">图像上的关键点位置</param>
		/// <param name="view_ix">从哪一个视图中进行估计</param>
		/// <returns>(左手有效性, 右手有效性)</returns>
		std::tuple<bool, bool> estimateHand(
			IN cv::Mat& img,
			OUT std::vector<cv::Point3f>& _kps_cam,
			OUT std::vector<cv::Point2f>& _kps_img,
			OUT std::optional<std::reference_wrapper<std::vector<cv::Rect2f>>> hand_bbox = std::nullopt,
			IN int view_ix = 0
		);

		/// <summary>
		/// 进行全身的姿态估计
		/// </summary>
		/// <param name="imgs">多视图图像</param>
		/// <param name="body_kps">人体关节点</param>
		/// <param name="hand_kps">手部关节点，前21个是人手的，后21个是人体的</param>
		/// <param name="hand_ref_view">从哪一个视图中进行手部估计</param>
		/// <returns>(全身有效性, 左手有效性, 右手有效性)</returns>
		std::tuple<bool, bool, bool> estimatePose(
			IN std::vector<cv::Mat>& imgs,
			OUT std::vector<cv::Point3f>& body_kps,
			OUT std::vector<cv::Point3f>& hand_kps,
			IN int hand_ref_view = 0
		);

		void estimatePose(
			IN std::vector<cv::Mat>& imgs,
			OUT PoseResult& pose_result,
			IN int hand_ref_view = 0
		);

	private:
		bool loadBodyModel();
		bool loadHandModel();

		std::vector<cv::Point3f> triangulate2DPoints(
			const std::vector<std::vector<cv::Point2f>>& img_coords
		);

	private:
		// 人体检测的 yolo 模型的地址和模型
		BodyModelConfig body_model_cfg;
		Yolov8Onnx body_model;
		// 人手检测的 HaMeR 模型的地址
		HandModelConfig hand_model_cfg;
		HaMeROnnx hand_model;
		// 使用的视图数量
		int num_views;
		// 相机的参数
		std::vector<CameraParameter> cameras;
	};

}