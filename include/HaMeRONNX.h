#pragma once

#include <tuple>
#include <string>
#include <opencv2/opencv.hpp>
#include <optional>
#include <onnxruntime_cxx_api.h>

#define IN
#define OUT

namespace BodyHand {

	class HaMeROnnx {
	public:
		HaMeROnnx();
		bool loadModel(std::string _handlr_model_path, std::string _hamer_model_path);
		std::tuple<bool, bool> detectPose(
			IN cv::Mat& img,
			IN const cv::Mat intr,
			IN const std::vector<float> undist,
			OUT std::vector<cv::Point3f>& hand_position_cam,
			OUT std::vector<cv::Point2f>& hand_position_2d,
			OUT std::optional<std::reference_wrapper<std::vector<cv::Rect2f>>> hand_bbox = std::nullopt
		);

	private:
		std::vector<float> postProcessHandPosition(std::vector<Ort::Value>& output_tensor, int lr_flag);
		std::tuple<std::vector<float>, std::vector<float>> postProcessHandRotation(std::vector<Ort::Value>& output_tensor, int lr_flag);
		std::vector<float> postProcessHandShape(std::vector<Ort::Value>& output_tensor, int lr_flag);
		std::vector<float> postProcessHand2D(std::vector<Ort::Value>& output_tensor, int lr_flag);
		std::tuple<
			std::vector<float>,
			std::vector<float>,
			std::vector<float>,
			std::vector<float>,
			std::vector<float>
		> postProcessOutputTensor(std::vector<Ort::Value>& output_hand_tensors, int lr_flag);
		std::tuple<std::vector<float>, std::vector<cv::Point3f>, std::vector<cv::Point2f>>
		restorePose(std::vector<float>& hand_position, std::vector<float>& hand_2d, int lr_flag, float scale_factor, cv::Point2f center);

		std::vector<cv::Point3f> recoverPoseCam(
			std::vector<cv::Point3f> local_pos,
			std::vector<cv::Point2f> img_coord,
			const cv::Mat& intr,
			const std::vector<float> undist
		);

		std::vector<Ort::Value> detectHandBox(cv::Mat& img);
		std::vector<Ort::Value> detectHandFromBox(cv::Mat& hand_img);
		std::vector<const char*> getInputNodeNames(Ort::Session* session);
		std::vector<const char*> getOutputNodeNames(Ort::Session* session);

	private:
		// ģ�͵�ַ
		std::string handlr_model_path;
		std::string hamer_model_path;
		// onnxģ�����л���
		Ort::Env ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "HaMeROnnx");
		Ort::SessionOptions ort_opt;
		Ort::Session* handlr_session;
		Ort::Session* hamer_session;
	};

}