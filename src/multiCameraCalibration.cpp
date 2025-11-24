#include <fstream>
#include <filesystem>
#include <opencv2/viz.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/calib3d.hpp>
#include "multiCameraCalibration.h"
#include "detectChessboardCorner.h"
#include "CameraController.h"
#include "multiCameraBA.h"


// 找出目录下所有包含view名称的子目录名的子目录，保存到列表中
bool findSubDirByViewName(const std::string& dirPath, const std::string& viewName, std::vector<std::string>& subDirList)
{
	namespace fs = std::filesystem;
	fs::path dir(dirPath);
	if (!fs::exists(dir) || !fs::is_directory(dir))
	{	
		std::cerr << "Error: " << dir << " is not a valid directory." << std::endl;
		return false;
	}

	for (const auto& entry : fs::directory_iterator(dir))
	{
		if (fs::is_directory(entry))
		{
			std::string subDirName = entry.path().filename().string();
			if (subDirName.find(viewName) != std::string::npos)
			{
				std::cout << "Found Camera Directory: " << subDirName << std::endl;
				subDirList.push_back(subDirName);
			}
		}
	}
	return true;
}

bool loadCornersFromFile(const std::string& filePath,
	std::vector<bool>& imagefound,
	std::vector<double>& _double_imagePoints,
	int cornersCountPerImage)
{
	std::ifstream file(filePath);
	if (!file.is_open())
	{
		std::cerr << "Error: open file " << filePath << " failed." << std::endl;
		return false;
	}

	int imageCount, imagefoundCount;
	file >> imageCount >> imagefoundCount;
	imagefound.resize(imageCount);
	_double_imagePoints.resize(imagefoundCount * cornersCountPerImage * 2);

	int temp; // file not support >> vector<bool>
	for (int i = 0; i < imageCount; i++)
	{
		file >> temp;
		imagefound[i] = (temp != 0);
	}
	for (int i = 0; i < imagefoundCount * cornersCountPerImage * 2; i++)
	{
		file >> _double_imagePoints[i];
	}
	file.close();
	return true;
}

bool convertDoubleToCVPts(const std::vector<double>& _double_imagePoints, std::vector<cv::Point2d>& imagePoints)
{
	imagePoints.resize(_double_imagePoints.size() / 2);
	for (size_t i = 0; i < imagePoints.size(); i++)
	{
		imagePoints[i].x = _double_imagePoints[i * 2];
		imagePoints[i].y = _double_imagePoints[i * 2 + 1];
	}
	return true;
}

bool findcommonImagePoints(std::vector<std::vector<cv::Point2d>>& imagePointsList,
	std::vector<std::vector<bool>>& imageFoundList,
	int cornersCountPerImage)
{
	using namespace std;
	size_t cameraNum = imageFoundList.size();
	size_t imageNumPerCam = imageFoundList[0].size();

	vector<bool> commonImageFound = imageFoundList[0];
	for (size_t i = 1; i < cameraNum; i++)
	{
		for (size_t j = 0; j < imageNumPerCam; j++)
		{
			commonImageFound[j] = commonImageFound[j] && imageFoundList[i][j];
		}
	}
	std::cout << "Common Image Found: " << std::count(commonImageFound.begin(), commonImageFound.end(), true) << std::endl;
	for (size_t i = 0; i < cameraNum; i++)
	{
		// two pointer to address it	
		size_t foundPointer = 0;
		size_t imagePointer = 0;
		while (foundPointer < imageNumPerCam)
		{
			if (imageFoundList[i][foundPointer] == 1)
			{
				if (commonImageFound[foundPointer] != 1)
				{
					//擦除前88个元素
					imagePointsList[i].erase(imagePointsList[i].begin() + imagePointer * cornersCountPerImage,
												imagePointsList[i].begin() + (imagePointer + 1) * cornersCountPerImage);
					foundPointer++;
				}
				else
				{
					imagePointer++;
					foundPointer++;
				}
			}
			else
			{
				foundPointer++;
			}
		}
	}
	return true;
}

template<typename T>
std::vector<cv::Mat> convertPointsToMats(const std::vector<std::vector<T>>& points2D) {
	std::vector<cv::Mat> mats;
	for (const auto& vec : points2D) {
		cv::Mat mat(vec); // 将 std::vector<cv::Point2f> 转换为 cv::Mat
		mats.push_back(mat);
	}
	return mats;
}

bool monoCalibration(CameraController& cam, int cornerCountPerImage, bool is_fix_instrinic = true) {
	size_t patternNum = cam.imagePoints.size()/cornerCountPerImage;
	std::vector<std::vector<cv::Point3f> > objectPoints(1);
	cv::Size boardSize = cam.boardSize;
	float squareSize = cam.squareSize;

	// generate chessboard 3f points
	for (int i = 0; i < boardSize.width; ++i) {
		for (int j = 0; j < boardSize.height; ++j) {
			objectPoints[0].push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
		}
	}
	objectPoints.resize(patternNum, objectPoints[0]); // extent to all view
	double rms;
	std::vector<std::vector<cv::Point2f>> inputimage2d(patternNum);
	for (int i = 0; i < patternNum; i++)
	{
		inputimage2d[i].reserve(cornerCountPerImage);
		for (int j = 0; j < cornerCountPerImage; j++) {
			inputimage2d[i].push_back(cv::Point2f(cam.imagePoints[i * cornerCountPerImage + j].x,
				cam.imagePoints[i * cornerCountPerImage + j].y));
		}
	}

	std::vector<cv::Mat> inputimage2dMat = convertPointsToMats(inputimage2d);
	std::vector<cv::Mat> objectPointsMat = convertPointsToMats(objectPoints);
	
	std::vector<cv::Mat> rvecs, tvecs;
	int flag = 0;
	if (is_fix_instrinic) {
		flag = cv::CALIB_FIX_ASPECT_RATIO
			| cv::CALIB_FIX_PRINCIPAL_POINT
			| cv::CALIB_USE_INTRINSIC_GUESS
			| cv::CALIB_ZERO_TANGENT_DIST
			| cv::CALIB_FIX_K1
			| cv::CALIB_FIX_K2
			| cv::CALIB_FIX_K3;
	}
	bool calibrateinstrinsic = true;
	if (calibrateinstrinsic) {
		//int flag = 0;
		rms = cv::calibrateCamera(
			objectPointsMat,
			inputimage2dMat,
			cam.imageSize,
			cam.cameraMatrix,
			cam.distCoeffs,
			rvecs,
			tvecs,
			flag
		);
	}
	
	//Rodrigues(rvecs[0], cam.R);
	//cam.T = tvecs[0];
	return true;
}

bool stereoCalibration(CameraController& firstCam,
	CameraController& secondCam,
	cv::Mat& R,
	cv::Mat& T,
	int cornerCountPerImage)
{
	size_t patternNum = firstCam.imagePoints.size()/cornerCountPerImage;
	cv::Mat E, F;
	cv::Mat perViewError;
	std::vector<std::vector<cv::Point3f> > objectPoints(1);
	cv::Size boardSize = firstCam.boardSize;
	float squareSize = firstCam.squareSize;

	// generate chessboard 3f points
	for (int i = 0; i < boardSize.width; ++i) {
		for (int j = 0; j < boardSize.height; ++j) {
			objectPoints[0].push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
		}
	}

	objectPoints.resize(patternNum, objectPoints[0]); // extent to all view

	//Find intrinsic and extrinsic camera parameters
	double rms;
	int iFixedPoint = -1;

	std::vector<std::vector<cv::Point2f>> firstInputimage2d(patternNum);
	for (int i = 0; i < patternNum; i++)
	{
		firstInputimage2d[i].reserve(cornerCountPerImage);
		for (int j = 0; j < cornerCountPerImage; j++) {
			firstInputimage2d[i].push_back(cv::Point2f(firstCam.imagePoints[i * cornerCountPerImage + j].x,
				firstCam.imagePoints[i * cornerCountPerImage + j].y));
		}
	}

	std::vector<std::vector<cv::Point2f>> secondInputimage2d(patternNum);
	for (int i = 0; i < patternNum; i++)
	{
		secondInputimage2d[i].reserve(cornerCountPerImage);
		for (int j = 0; j < cornerCountPerImage; j++) {
			secondInputimage2d[i].push_back(cv::Point2f(secondCam.imagePoints[i * cornerCountPerImage + j].x,
				secondCam.imagePoints[i * cornerCountPerImage + j].y));
		}
	}

	std::vector<cv::Mat> rvecs, tvecs;
	std::vector<cv::Mat> firstInputimage2dMat = convertPointsToMats(firstInputimage2d);
	std::vector<cv::Mat> secondInputimage2dMat = convertPointsToMats(secondInputimage2d);
	std::vector<cv::Mat> objectPointsMat = convertPointsToMats(objectPoints);
	
	int flag =   cv::CALIB_FIX_INTRINSIC | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_K3;
	cv::stereoCalibrate(
		objectPoints,
		firstInputimage2d,
		secondInputimage2d,
		firstCam.cameraMatrix,
		firstCam.distCoeffs,
		secondCam.cameraMatrix,
		secondCam.distCoeffs,
		firstCam.imageSize,
		R,
		T,
		E,
		F,
		perViewError,
		flag
	);
	//std::cout << "first camera matrix: " << firstCam.cameraMatrix << std::endl;
	//std::cout << "second camera matrix: " << secondCam.cameraMatrix << std::endl;
	//std::cout << "distortion1 coefficients: " << firstCam.distCoeffs << std::endl;
	//std::cout << "distortion2 coefficients: " << secondCam.distCoeffs << std::endl;
	//std::cout << "per view error: " << perViewError << std::endl;
	return 0;
}

double computeReprojectionError(
	const std::vector<cv::Point2d>& imagePoints1,
	const std::vector<cv::Point2d>& imagePoints2,
	const cv::Mat& R,
	const cv::Mat& T,
	const cv::Mat& K1,
	const cv::Mat& K2,
	const cv::Mat& points3D
) {
	using namespace cv;
	using namespace std;
	double totalError = 0.0;
	int pointCount = 0;
	Mat projMat1 = K1 * Mat::eye(3, 4, R.type());
	Mat projMat2 = K2 * (Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2));

	for (int i = 0; i < points3D.rows; i++) {
		Point3d point3D = Point3d(points3D.at<double>(i, 0),
			points3D.at<double>(i, 1),
			points3D.at<double>(i, 2));

		// Project 3D points back to image plane
		Mat pointMat = (Mat_<double>(4, 1) << point3D.x, point3D.y, point3D.z, 1.0);
		Mat projected1 = projMat1 * pointMat;
		Mat projected2 = projMat2 * pointMat;

		Point2d projectedPt1(
			projected1.at<double>(0, 0) / projected1.at<double>(2, 0),
			projected1.at<double>(1, 0) / projected1.at<double>(2, 0)
		);

		Point2d projectedPt2(
			projected2.at<double>(0, 0) / projected2.at<double>(2, 0),
			projected2.at<double>(1, 0) / projected2.at<double>(2, 0)
		);

		// Calculate the error
		double error1 = norm(imagePoints1[i] - projectedPt1);
		double error2 = norm(imagePoints2[i] - projectedPt2);

		totalError += error1 + error2;
		pointCount += 2;
	}

	// Return the average reprojection error
	return totalError / pointCount;
}


std::vector<cv::Point3d> convertMatToPoint3d(const cv::Mat& points3d) {
	std::vector<cv::Point3d> points;

	if (points3d.rows == 3 && points3d.channels() == 1) {
		// 处理 3 x N 单通道矩阵
		points.reserve(points3d.cols); // 预分配空间以提高性能

		for (int i = 0; i < points3d.cols; ++i) {
			// 访问 3 x N 矩阵中的每个点
			double x = points3d.at<double>(0, i);
			double y = points3d.at<double>(1, i);
			double z = points3d.at<double>(2, i);
			points.push_back(cv::Point3d(x, y, z));
		}
	}
	else if (points3d.rows == 1 && points3d.channels() == 3) {
		// 处理 1 x N 三通道矩阵
		points.reserve(points3d.cols); // 预分配空间以提高性能

		for (int i = 0; i < points3d.cols; ++i) {
			// 访问 1 x N 矩阵中的每个点
			cv::Vec3d vec = points3d.at<cv::Vec3d>(0, i);
			points.push_back(cv::Point3d(vec[0], vec[1], vec[2]));
		}
	}
	else if (points3d.rows == points3d.total() && points3d.channels() == 3 && points3d.cols == 1) {
		// 处理 N x 1 三通道矩阵
		points.reserve(points3d.rows); // 预分配空间以提高性能

		for (int i = 0; i < points3d.rows; ++i) {
			// 访问 N x 1 矩阵中的每个点
			cv::Vec3d vec = points3d.at<cv::Vec3d>(i, 0);
			points.push_back(cv::Point3d(vec[0], vec[1], vec[2]));
		}
	}
	else {
		std::cerr << "Unsupported matrix format" << std::endl;
	}

	return points;
}

double computeReprojectionErrors(const std::vector<cv::Point3d>& objectPoints,
	const std::vector<cv::Point2d>& imagePoints,
	const cv::Mat& rvecs,
	const cv::Mat& tvecs,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs)
{
	std::vector<cv::Point2d> imagePoints2;
	double err = 0;

	projectPoints(objectPoints, rvecs, tvecs, cameraMatrix, distCoeffs, imagePoints2);
	err = norm(imagePoints, imagePoints2, cv::NORM_L2);
	size_t n = objectPoints.size();
	err = (double)std::sqrt(err * err / n);
	return err;
}

cv::Mat convert2channelTo2xN(const cv::Mat& points) {
	// 检查输入矩阵的大小和通道数
	CV_Assert(points.channels() == 2 && (points.rows == 1 || points.cols == 1));

	// 分离通道
	std::vector<cv::Mat> channels(2);
	cv::split(points, channels);

	// 创建 2xN 的矩阵
	cv::Mat result(2, points.total(), CV_64F);

	// 将分离的通道数据复制到新矩阵
	channels[0].reshape(1, 1).copyTo(result.row(0)); // 第一个通道
	channels[1].reshape(1, 1).copyTo(result.row(1)); // 第二个通道

	return result;
}

double computeReprojectionError(const std::vector<cv::Point3d>& objectPoints,
	const std::vector<cv::Point2d>& imagePoints,
	const cv::Mat& cameraMatrix,
	const cv::Mat& distCoeffs,
	const cv::Mat& R,
	const cv::Mat& tvec) {
	// 将旋转矩阵转换为旋转向量
	cv::Mat rvec;
	cv::Rodrigues(R, rvec);

	std::vector<cv::Point2d> projectedPoints;
	cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

	double totalError = 0;
	for (size_t i = 0; i < imagePoints.size(); ++i) {
		double error = cv::norm(imagePoints[i] - projectedPoints[i]);
		totalError += error * error;
	}

	return std::sqrt(totalError / imagePoints.size());
}

void printMatColumns(const cv::Mat& mat, int numCols) {
	// 检查输入的列数是否超过矩阵的总列数
	if (numCols > mat.cols) {
		numCols = mat.cols;
	}

	// 选择前 numCols 列的子矩阵
	cv::Mat subMat = mat(cv::Rect(0, 0, numCols, mat.rows));

	// 输出子矩阵
	//std::cout << "Sub matrix (" << numCols << " columns):" << std::endl;
	//std::cout << subMat << std::endl;
}

std::vector<cv::Point2f> convertPoints2dToPoints2f(const std::vector<cv::Point2d>& points2d) {
	std::vector<cv::Point2f> points2f;
	points2f.reserve(points2d.size()); // 预分配空间以提高性能

	for (const auto& point : points2d) {
		points2f.push_back(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)));
	}

	return points2f;
}

std::vector<cv::Point2d> convertPoints2fToPoints2d(const std::vector<cv::Point2f>& points2f) {
	std::vector<cv::Point2d> points2d;
	points2d.reserve(points2f.size()); // 预分配空间以提高性能

	for (const auto& point : points2f) {
		points2d.push_back(cv::Point2d(static_cast<double>(point.x), static_cast<double>(point.y)));
	}

	return points2d;
}

bool PaperCalibration_pnp(std::vector<CameraController>& cameraList,
	std::vector<cv::Point3d>& worldPoints,
	int cornerCountsPerImage, bool is_fix_instrinic = true)
{
	using namespace std;
	using namespace cv;
	// calibrate mono cam
	//for (size_t i = 0; i < cameraList.size(); i++)
	//{
	//	CameraController& camera = cameraList[i];
	//	monoCalibration(camera, cornerCountsPerImage, is_fix_instrinic);
	//}

	size_t firstindex = 0;
	size_t secondindex = 1;
	CameraController& firstCam = cameraList[firstindex];
	CameraController& secondCam = cameraList[secondindex];

	cv::Mat R, T;

	stereoCalibration(firstCam, secondCam, R, T, cornerCountsPerImage);
	secondCam.R = R;
	secondCam.T = T;
	
	std::vector<CameraController*> cameras;
	std::vector<std::vector<cv::Point2d>> imagePoints;
	
	std::vector<cv::Point2f> undistortedImagePoints1f;
	std::vector<cv::Point2f> undistortedImagePoints2f;
	cv::undistortImagePoints(convertPoints2dToPoints2f(firstCam.imagePoints), undistortedImagePoints1f, firstCam.cameraMatrix, firstCam.distCoeffs);
	cv::undistortImagePoints(convertPoints2dToPoints2f(secondCam.imagePoints), undistortedImagePoints2f, secondCam.cameraMatrix, secondCam.distCoeffs);
	std::vector<cv::Point2d> undistortedImagePoints1;
	std::vector<cv::Point2d> undistortedImagePoints2;
	undistortedImagePoints1 = convertPoints2fToPoints2d(undistortedImagePoints1f);
	undistortedImagePoints2 = convertPoints2fToPoints2d(undistortedImagePoints2f);

	Mat points4D;
	Mat P1 = firstCam.ProjectMatrix();
	Mat P2 = secondCam.ProjectMatrix();
	triangulatePoints(P1, P2,
		undistortedImagePoints1, undistortedImagePoints2, points4D);

	Mat points3D;
	convertPointsFromHomogeneous(points4D.t(), points3D);
	worldPoints = convertMatToPoint3d(points3D);
	double rms1 = computeReprojectionError(worldPoints, firstCam.imagePoints, firstCam.cameraMatrix, firstCam.distCoeffs, firstCam.R, firstCam.T);
	double rms2 = computeReprojectionError(worldPoints, secondCam.imagePoints, secondCam.cameraMatrix, secondCam.distCoeffs, secondCam.R, secondCam.T);
	
	std::cout << "triangulate reprojection error on view 1: " << rms1 << std::endl;
	std::cout << "triangulate reprojection error on view 2: " << rms2 << std::endl;

	cameras.push_back(&firstCam);
	cameras.push_back(&secondCam);

	std::cout << firstCam.distCoeffs << std::endl;
	std::cout << secondCam.distCoeffs << std::endl;

	imagePoints.push_back(firstCam.imagePoints);
	imagePoints.push_back(secondCam.imagePoints);

	//OptimizeCameraAndPoints(cameras, worldPoints, imagePoints);
	for (int i = 0; i < cameraList.size(); i++)
	{

		if (i == firstindex || i == secondindex) continue;
		CameraController& camera = cameraList[i];

		Mat inlier;
		Mat rvec, tvec;
		solvePnPRansac(worldPoints,
			camera.imagePoints,
			camera.cameraMatrix,
			camera.distCoeffs,
			rvec,
			tvec,
			false,
			100,
			0.9,
			0.99,
			inlier,
			SOLVEPNP_EPNP
		);

		Rodrigues(rvec, camera.R);
		camera.T = tvec;

		//BA
		cameras.push_back(&camera);
		imagePoints.push_back(camera.imagePoints);
	}

	double meanE = 0;
	for (size_t i = 0; i < cameraList.size(); i++)
	{
		CameraController& camera = cameraList[i];	
		double error = computeReprojectionError(worldPoints,
			camera.imagePoints,
			camera.cameraMatrix,
			camera.distCoeffs,
			camera.R, camera.T);
		std::cout << "camera " << i << " reprojection error: " << error << std::endl;	
		meanE += error;
	}
	std::cout << "origin mean reprojection error: " << meanE / cameraList.size() << std::endl;
	OptimizeCameraAndPoints(cameras, worldPoints, imagePoints);
	meanE = 0;	
	for (size_t i = 0; i < cameraList.size(); i++)
	{
		CameraController& camera = cameraList[i];	
		double error = computeReprojectionError(worldPoints,
			camera.imagePoints,
			camera.cameraMatrix,
			camera.distCoeffs,
			camera.R, camera.T);
		std::cout << "camera " << i << " reprojection error: " << error << std::endl;	
		meanE += error;
	}

	std::cout << "optimal mean reprojection error: " << meanE / cameraList.size() << std::endl;
	return true;
}

bool PaperCalibration_twostereo(std::vector<CameraController>& cameraList,
	std::vector<cv::Point3d>& worldPoints,
	int cornerCountsPerImage)
{
	using namespace std;
	using namespace cv;

	for (size_t i = 0; i < cameraList.size(); i++)
	{
		CameraController& camera = cameraList[i];
		monoCalibration(camera, cornerCountsPerImage);
	}

	size_t mainindex = 0;
	for (size_t i = 0; i < cameraList.size(); i++)
	{
		if (i == mainindex) continue;
		CameraController& firstCam = cameraList[mainindex];
		CameraController& secondCam = cameraList[i];

		cv::Mat R, T;

		stereoCalibration(firstCam, secondCam, R, T, cornerCountsPerImage);
		secondCam.R = R;
		secondCam.T = T;
	}
	
	std::vector<CameraController*> cameras;
	std::vector<std::vector<cv::Point2d>> imagePoints;
	std::vector<cv::Mat> points2d;
	std::vector<cv::Mat> projection_matrices;
	for (int i = 0; i < cameraList.size(); i++)
	{
		CameraController& camera = cameraList[i];
		cameras.push_back(&camera);
		imagePoints.push_back(camera.imagePoints);
		projection_matrices.push_back(camera.ProjectMatrix());
		cv::Mat undistortedImagePoints;
		cv::undistortImagePoints(camera.convertPointsToMat(), undistortedImagePoints, camera.cameraMatrix, camera.distCoeffs);
		//printMatColumns(camera.convertPointsToMat(), 8);
		//printMatColumns(convert2channelTo2xN(undistortedImagePoints), 8);
		points2d.push_back(convert2channelTo2xN(undistortedImagePoints));
	}

	//Triangulate points
	cv::Mat points3d;
	cv::sfm::triangulatePoints(points2d, projection_matrices, points3d);

	worldPoints = convertMatToPoint3d(points3d);

	double meanE = 0;
	for (size_t i = 0; i < cameraList.size(); i++)
	{
		CameraController& camera = cameraList[i];	
		std::cout << "cameraMatrix: " << camera.cameraMatrix << std::endl;
		std::cout << "distCoeffs: " << camera.distCoeffs << std::endl;
		std::cout << "R: " << camera.R << std::endl;
		std::cout << "T: " << camera.T << std::endl;
		double error = computeReprojectionError(worldPoints,
			camera.imagePoints,
			camera.cameraMatrix,
			camera.distCoeffs,
			camera.R, camera.T);
		std::cout << "camera " << i << " reprojection error: " << error << std::endl;	
		meanE += error;
	}
	std::cout << "mean reprojection error: " << meanE / cameraList.size() << std::endl;
	OptimizeCameraAndPoints(cameras, worldPoints, imagePoints);
	meanE = 0;
	for (size_t i = 0; i < cameraList.size(); i++){
		CameraController& camera = cameraList[i];	
		double error = computeReprojectionError(worldPoints,
			camera.imagePoints,
			camera.cameraMatrix,
			camera.distCoeffs,
			camera.R, camera.T);
		std::cout << "camera " << i << " reprojection error: " << error << std::endl;	
		meanE += error;
	}
	std::cout << "mean reprojection error: " << meanE / cameraList.size() << std::endl;
	return true;
}

bool zhangCalibration(std::vector<CameraController>& cameraList,
	std::vector<cv::Point3d>& worldPoints,
	int cornerCountsPerImage)
{
	using namespace std;
	using namespace cv;
	size_t firstindex = 3;
	size_t secondindex = 4;
	
	size_t imageNumPerImage = cameraList[0].imagePoints.size();
	CameraController& firstCam = cameraList[firstindex];
	CameraController& secondCam = cameraList[secondindex];

	cv::Mat R, T;
	cv::Mat K;
	K = firstCam.cameraMatrix;

	// stereo calibration input parameter
	stereoCalibration(firstCam, secondCam, R, T, cornerCountsPerImage);

	secondCam.R = R;
	secondCam.T = T;

	Mat points4D;
	// 投影矩阵 P1 (相机1) 假设是原点处的相机
	Mat P1 = firstCam.cameraMatrix * cv::Mat::eye(3, 4, CV_64F); // K * [I | 0]

	// 投影矩阵 P2 (相机2) K * [R | T]
	Mat P2 = secondCam.cameraMatrix * (cv::Mat_<double>(3, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2));
	triangulatePoints(P1, P2,
		firstCam.imagePoints, secondCam.imagePoints, points4D);

	Mat points3D;
	convertPointsFromHomogeneous(points4D.t(), points3D);

	double reprojectionError = computeReprojectionError(firstCam.imagePoints,
		secondCam.imagePoints, R, T, firstCam.cameraMatrix, secondCam.cameraMatrix, points3D);
	
	std::cout << "triangulate reprojection error: " << reprojectionError << std::endl;
	worldPoints = convertMatToPoint3d(points3D);

	std::vector<CameraController*> cameras;
	cameras.push_back(&firstCam);
	cameras.push_back(&secondCam);

	std::vector<std::vector<cv::Point2d>> imagePoints;
	imagePoints.push_back(firstCam.imagePoints);
	imagePoints.push_back(secondCam.imagePoints);

	//OptimizeCameraAndPoints(cameras, worldPoints, imagePoints);
	for (int i = 0; i < cameraList.size(); i++)
	{

		if (i == firstindex || i == secondindex) continue;
		CameraController& camera = cameraList[i];

		Mat inlier;
		Mat rvec, tvec;
		solvePnPRansac(worldPoints,
			camera.imagePoints,
			camera.cameraMatrix,
			camera.distCoeffs,
			rvec,
			tvec,
			false,
			100,
			0.9,
			0.99,
			inlier,
			SOLVEPNP_EPNP
		);

		double error = computeReprojectionErrors(worldPoints,
			camera.imagePoints,
			rvec,
			tvec,
			camera.cameraMatrix,
			camera.distCoeffs);

		Rodrigues(rvec, camera.R);
		camera.T = tvec;

		 //BA
		cameras.push_back(&camera);
		imagePoints.push_back(camera.imagePoints);
		//if ( i == (cameraList.size()-1 ) )
			//OptimizeCameraAndPoints(cameras, worldPoints, imagePoints);
	}
	double meanE = 0;
	for (size_t i = 0; i < cameraList.size(); i++)
	{
		CameraController& camera = cameraList[i];	
		cv::Mat rvec;
		Rodrigues(camera.R, rvec);
		double error = computeReprojectionErrors(worldPoints,
			camera.imagePoints,
			rvec,
			camera.T,
			camera.cameraMatrix,
			camera.distCoeffs);
		std::cout << "camera " << i << " reprojection error: " << error << std::endl;	
		meanE += error;
	}
	std::cout << "mean reprojection error: " << meanE / cameraList.size() << std::endl;
	return true;
}

bool viz(std::vector<CameraController>& cameraMatrix, std::vector<cv::Point3d>& worldPoints)
{
	// 创建一个viz窗口
	using namespace cv;
	cv::viz::Viz3d window("Camera Poses Visualization");

	// 设置背景颜色为白色
	window.setBackgroundColor(cv::viz::Color::white());

	unsigned int camNum = cameraMatrix.size();
	for (size_t i = 0; i < camNum; i++) {
		CameraController& camera = cameraMatrix[i];
		cv::Mat rotation;
		rotation = camera.R;

		// 构建4x4的仿射矩阵
		cv::Mat affine = cv::Mat::zeros(4, 4, CV_64F);
		rotation.copyTo(affine(cv::Rect(0, 0, 3, 3)));
		cv::Mat C = -camera.R.inv() * camera.T;
		C.copyTo(affine(cv::Rect(3, 0, 1, 3)));
		affine.at<double>(3, 3) = 1.0;
		Affine3d pose(affine);

		// 创建一个小立方体来表示相机的体积
		Matx33d K(camera.cameraMatrix);
		viz::WCameraPosition cameraModel(40);
		window.showWidget("Cube" + std::to_string(i), cameraModel, pose);

		// 创建棋盘格角点方向射线
		cv::Point2d pt = camera.imagePoints[0];
		cv::Mat p_img_hom = (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1);
		cv::Mat p_norm = K.inv() * p_img_hom;
		cv::Mat st = C;
		cv::Mat et = camera.R.inv() * p_norm;
		Point3d startPt(st);
		Point3d endPt(et);
		window.showWidget("Pt" + std::to_string(i), cv::viz::WLine(startPt, startPt + endPt * 1000, cv::viz::Color::red()));

		// 显示编号
		std::string cameraNumber = "C " + std::to_string(i + 1);
		viz::WText3D cameraLabel(cameraNumber, Point3d(0, -0.1, 0), 10, false, viz::Color::white());
		window.showWidget("Label" + std::to_string(i), cameraLabel, pose);
	}

	// 定义颜色列表
    std::vector<cv::viz::Color> colors = {
        cv::viz::Color::red(), cv::viz::Color::green(), cv::viz::Color::blue(),
        cv::viz::Color::yellow(), cv::viz::Color::cyan(), cv::viz::Color::magenta()
    };

	// 假设每88个点是一个8x11的棋盘格
	size_t points_per_grid = NUM_WIDTH * NUM_HEIGHT;
	size_t rows = NUM_HEIGHT;
	size_t cols = NUM_WIDTH;

	// 只显示第一个棋盘格
	cv::viz::Color color = colors[0];

	// 绘制行线
	for (size_t row = 0; row < rows; ++row) {
		for (size_t col = 0; col < cols - 1; ++col) {
			size_t idx1 = row + col * rows;
			size_t idx2 = idx1 + rows;
			if (idx2 < worldPoints.size()) {
				cv::Point3f pt1(worldPoints[idx1].x, worldPoints[idx1].y, worldPoints[idx1].z);
				cv::Point3f pt2(worldPoints[idx2].x, worldPoints[idx2].y, worldPoints[idx2].z);
				cv::viz::WLine line(pt1, pt2, color);
				window.showWidget("line_row_" + std::to_string(idx1), line);
			}
		}
	}

	// 绘制列线
	for (size_t col = 0; col < cols; ++col) {
		for (size_t row = 0; row < rows - 1; ++row) {
			size_t idx1 = col * rows + row;
			size_t idx2 = idx1 + 1;
			if (idx2 < worldPoints.size()) {
				cv::Point3f pt1(worldPoints[idx1].x, worldPoints[idx1].y, worldPoints[idx1].z);
				cv::Point3f pt2(worldPoints[idx2].x, worldPoints[idx2].y, worldPoints[idx2].z);
				cv::viz::WLine line(pt1, pt2, color);
				window.showWidget("line_col_" + std::to_string(idx1), line);
			}
		}
	}


	window.spin();
	return true;
}


bool SaveCameraControllerToFile(std::string saveFilePath,
	std::vector<CameraController>& cameraList,
	std::vector<cv::Point3d>& worldPoints)
{
	std::ofstream ofs(saveFilePath);
	if (!ofs.is_open())
	{
		std::cerr << "Cannot open file: " << saveFilePath << std::endl;
		return false;
	}

	int cameraCount = cameraList.size();
	ofs << cameraCount << std::endl;

	for (const auto& camera : cameraList)
	{
		ofs << camera.cameraMatrix << std::endl;
		ofs << camera.distCoeffs << std::endl;
		ofs << camera.R << std::endl;
		ofs << camera.T << std::endl;
		cv::Mat ProjectionMatrix = cv::Mat::zeros(3, 4, CV_64F);
		camera.R.copyTo(ProjectionMatrix(cv::Rect(0, 0, 3, 3)));
		camera.T.copyTo(ProjectionMatrix(cv::Rect(3, 0, 1, 3)));

		ofs << ProjectionMatrix << std::endl;
	}

	//for (const auto& pt : worldPoints)
	//{
	//	ofs << pt.x << " " << pt.y << " " << pt.z << std::endl;
	//}
	ofs.close();	
	return true;
}

bool multiCameraCalibrationAndSave(std::string& calibrationDir,
	std::string& viewName, int cornersCountPerImage, bool is_fix_instrinic = true)
{
	using namespace std;
	bool result;
	vector<CameraController> cameraList;
	vector<string> subDirList;

	result = findSubDirByViewName(calibrationDir, viewName, subDirList);
	if (!result)
	{
		cerr << "Error: findSubDirByViewName failed." << endl;
		return false;
	}

	// 提取角点
	vector<vector<cv::Point2d>> imagePointsList;
	vector<vector<bool>> imageFoundList;
	for (const auto& subDir : subDirList)
	{
		std::string imageDirPath = calibrationDir + "/" + subDir;
		std::string cornertextPath = imageDirPath + "/" + "corners.txt";

		// 检查是否已经提取过角点
		if (!std::filesystem::exists(cornertextPath))
		//if (1)
		{
			result = detectMultiImageFileCorners(imageDirPath, cornersCountPerImage);
			if(!result)
			{
				std::cerr << "Error: detectMultiImageFileCorners failed. The camera dir is " 
					<< subDir << std::endl;
			}
		}
		vector<bool> imagefound;
		vector<double> _double_imagePoints;
		vector<cv::Point2d> imagePoints;
		result = loadCornersFromFile(cornertextPath, imagefound, _double_imagePoints, cornersCountPerImage);
		result = convertDoubleToCVPts(_double_imagePoints, imagePoints);
		imageFoundList.insert(imageFoundList.end(), imagefound);
		imagePointsList.insert(imagePointsList.end(), imagePoints);
	}
	// 找到都检测角点的视图
	findcommonImagePoints(imagePointsList, imageFoundList, cornersCountPerImage);

	// 添加相机 到 控制器
	for (size_t i = 0; i < imagePointsList.size(); i++)
	{
		CameraController camera;
		camera.imagePoints = imagePointsList[i];
		cameraList.push_back(camera);
	}
		
	// 多相机标定
	std::vector<cv::Point3d> worldPoints;
	std::cout << "calibration dir: " << calibrationDir << std::endl;
	PaperCalibration_pnp(cameraList, worldPoints, cornersCountPerImage, is_fix_instrinic);	
	
	SaveCameraControllerToFile(calibrationDir + "/" + "camera_param.txt", cameraList, worldPoints);

	viz(cameraList, worldPoints);
}
