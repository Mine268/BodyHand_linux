#pragma once 
#include <opencv2/opencv.hpp>
#include "constants.h"

class CameraController
{
public:
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	cv::Mat R;
	cv::Mat T;	

	std::vector<cv::Point2d> imagePoints;

	cv::Size boardSize;
	cv::Size imageSize;
	double squareSize;

	void setCameraMatrix(cv::Mat Matrix)
	{
		cameraMatrix = Matrix;
	}

	void setDistCoeffs(cv::Mat Coeffs)
	{
		distCoeffs = Coeffs;
	}

	void setRotationVecs(cv::Mat Rotation)
	{
		R = Rotation;
	}

	void setTranslationVecs(cv::Mat Translation)
	{
		T = Translation;
	}



	CameraController()
	{
		// camera parameter
		cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
		cameraMatrix.at<double>(0, 0) = 1045.0;
		cameraMatrix.at<double>(1, 1) = 1045.0;
		cameraMatrix.at<double>(0, 2) = 720.0;
		cameraMatrix.at<double>(1, 2) = 540.0;
		cameraMatrix.at<double>(2, 2) = 1;

		distCoeffs = (cv::Mat_<double>(1, 5) << -0.0751, -0.1446, 0, 0, 0.2794);
		R = cv::Mat::eye(3, 3, CV_64F);
		T = cv::Mat::zeros(3, 1, CV_64F);

		// chessboard parameter
		boardSize = cv::Size(NUM_WIDTH, NUM_HEIGHT);
		squareSize = SQUARE_SIZE;
		imageSize = cv::Size(1440, 1080);
	}

	cv::Mat ProjectMatrix()
	{
		cv::Mat projectMatrix = cv::Mat::zeros(3, 4, CV_64F);

		// 计算投影矩阵 P = K [R | T]
		cv::Mat RT(3, 4, CV_64F);
		R.copyTo(RT(cv::Rect(0, 0, 3, 3))); // 复制 R 到 RT 的前 3 列
		T.copyTo(RT(cv::Rect(3, 0, 1, 3))); // 复制 T 到 RT 的第 4 列

		projectMatrix = cameraMatrix * RT; // 计算投影矩阵

		return projectMatrix;
	}

	cv::Mat convertPointsToMat() {
		cv::Mat mat(2, imagePoints.size(), CV_64F);
		for (size_t i = 0; i < imagePoints.size(); ++i) {
			mat.at<double>(0, i) = imagePoints[i].x;
			mat.at<double>(1, i) = imagePoints[i].y;
		}
		return mat;
	}
};
