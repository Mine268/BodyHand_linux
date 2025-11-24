#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/viz.hpp>
#include "CameraController.h"

void readMat(std::ifstream& fin, cv::Mat& mat, int rows, int cols) {
	char nextc;
	double value;
	int col = 0;
	int row = 0;
	while(true)
	{
		nextc = fin.peek();
		if (fin.peek() == ',' || fin.peek() == ';' || 
			fin.peek() == '[' || fin.peek() == ']' ||
			fin.peek() == '\n' || fin.peek() == ' ') fin.ignore();
		else
		{
			fin >> value;
			if (cols == 1) mat.at<double>(row) = value;
			else mat.at<double>(col, row) = value;	
			if(row == (rows -1) && col == (cols - 1)) break;
			else if (row == (rows - 1))
			{
				row = 0;
				col++;
			}
			else row++;
		}
	}
}

bool viz(std::vector<CameraController>& cameraMatrix)
{
	using namespace cv;
	// 创建一个viz窗口
	cv::viz::Viz3d window("Camera Poses Visualization");

	size_t camNum = cameraMatrix.size();
	for (size_t i = 0; i < camNum; i++) {
		CameraController& camera = cameraMatrix[i];

		// 构建4x4的仿射矩阵
		cv::Mat affine = cv::Mat::zeros(4, 4, CV_64F);
		camera.R.copyTo(affine(cv::Rect(0, 0, 3, 3)));
		cv::Mat C = -camera.R.inv() * camera.T;
		C.copyTo(affine(cv::Rect(3, 1, 1, 3)));
		affine.at<double>(3, 3) = 1.0;
		Affine3d pose(affine);

		// 创建一个小立方体来表示相机的体积
		Matx33d K(camera.cameraMatrix);
		viz::WCameraPosition cameraModel(K, 10, viz::Color::white());
		window.showWidget("Cube" + std::to_string(i), cameraModel, pose);


		// 显示编号
		std::string cameraNumber = "C " + std::to_string(i + 1);
		viz::WText3D cameraLabel(cameraNumber, Point3d(0, -0.1, 0), 10, false, viz::Color::white());
		window.showWidget("Label" + std::to_string(i), cameraLabel, pose);
	}

	window.spin();
	return true;
}

bool visualizeCameraCalibrationResult(const std::string& calibrationResultFile)
{
	std::ifstream fin(calibrationResultFile);
	if (!fin.is_open())
	{
		std::cerr << "can not open file " << calibrationResultFile << std::endl;
		return false;
	}

	int cameraCount;
	fin >> cameraCount;
	std::vector<CameraController> cameras(cameraCount);
	for (CameraController& cam : cameras) {
		readMat(fin, cam.cameraMatrix, 3, 3);
		readMat(fin, cam.distCoeffs, 5, 1);
		readMat(fin, cam.R, 3, 3);
		readMat(fin, cam.T, 3, 1);
	}

	fin.close();

	viz(cameras);

	return true;
}

bool printProjectMatrix(const std::string& calibrationResultFile)
{
	std::ifstream fin(calibrationResultFile);
	if (!fin.is_open())
	{
		std::cerr << "can not open file " << calibrationResultFile << std::endl;
		return false;
	}

	int cameraCount;
	fin >> cameraCount;
	std::vector<CameraController> cameras(cameraCount);
	for (int i = 0; i < cameraCount; i++) {
		CameraController& cam = cameras[i];
		readMat(fin, cam.cameraMatrix, 3, 3);
		readMat(fin, cam.distCoeffs, 5, 1);
		readMat(fin, cam.R, 3, 3);
		readMat(fin, cam.T, 3, 1);
		cv::Mat ExtrinsicMatrix;
		cv::hconcat(cam.R, cam.T, ExtrinsicMatrix);

		//cv::Mat ProjectionMatrix = cam.cameraMatrix * ExtrinsicMatrix;
		//std::cout << "Camera " << i << ": \n" << ProjectionMatrix << std::endl;

		std::cout << "Camera " << i << ": \n" << ExtrinsicMatrix << std::endl;
	}

	
}
