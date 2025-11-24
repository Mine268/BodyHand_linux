#pragma once 
#include <opencv2/opencv.hpp>
#include <vector>
#include "CameraController.h"

void OptimizeCameraAndPoints(
	std::vector<CameraController*>& cameraMatrix,
	std::vector<cv::Point3d>& worldPoints,
	std::vector<std::vector<cv::Point2d>>& imagePoints);
