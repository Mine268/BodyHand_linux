#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "multiCameraBA.h"

// 用于Ceres的重投影误差计算，包括畸变处理
struct ReprojectionErrorWithDistortion {
	cv::Point2d observed;

	ReprojectionErrorWithDistortion(cv::Point2d observed)
		: observed(observed) {}

	template <typename T>
	bool operator()(const T* const camera_r, // 相机姿态参数
		const T* const camera_t,
		const T* const point, // 3D点坐标
		const T* const fx,
		const T* const cx, 
		const T* const alpha, 
		const T* const cy, 
		const T* const distortion_coeffs, // 畸变系数
		T* residuals) const {
		// 相机姿态转换和3D点投影到相机坐标系
		T p[3];
		ceres::AngleAxisRotatePoint(camera_r, point, p);
		p[0] += camera_t[0]; p[1] += camera_t[1]; p[2] += camera_t[2];

		// 归一化坐标
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// 应用径向畸变
		T r2 = xp * xp + yp * yp;
		T radial_distortion = T(1) + distortion_coeffs[0] * r2 + distortion_coeffs[1] * r2 * r2 + distortion_coeffs[4] * r2 * r2 * r2;
		T xpp = xp * radial_distortion + T(2) * distortion_coeffs[2] * xp * yp + distortion_coeffs[3] * (r2 + T(2) * xp * xp);
		T ypp = yp * radial_distortion + distortion_coeffs[2] * (r2 + T(2) * yp * yp) + T(2) * distortion_coeffs[3] * xp * yp;

		// 应用内参矩阵
		//const T& fx = intrinsic[0];
		//const T& fy = intrinsic[4];
		//const T& cx = intrinsic[2];
		//const T& cy = intrinsic[5];

		T predicted_x = fx[0] * xpp + cx[0];
		T predicted_y = fx[0]*alpha[0] * ypp + cy[0];

		// 计算误差
		residuals[0] = predicted_x - T(observed.x);
		residuals[1] = predicted_y - T(observed.y);
		return true;
	}
};

bool CheckTypesForOptimization(
	const std::vector<CameraController*>& cameras,
	const std::vector<cv::Point3d>& worldPoints,
	const std::vector<std::vector<cv::Point2d>>& imagePoints) {

	// 检查相机参数类型
	for (const auto& camera : cameras) {
		if (camera->R.type() != CV_64F || camera->T.type() != CV_64F ||
			camera->cameraMatrix.type() != CV_64F || camera->distCoeffs.type() != CV_64F) {
			std::cerr << "Error: Camera parameters must be of type CV_64F." << std::endl;
			return false;
		}
		if (camera->R.empty() || camera->T.empty() ||
			camera->cameraMatrix.empty() || camera->distCoeffs.empty()) {
			std::cerr << "Error: Camera parameters must not be empty." << std::endl;
			return false;
		}
	}

	// 检查世界点的大小
	if (worldPoints.empty()) {
		std::cerr << "Error: World points must not be empty." << std::endl;
		return false;
	}

	// 检查图像点的大小和类型
	if (imagePoints.size() != cameras.size()) {
		std::cerr << "Error: The number of image points vectors must match the number of cameras." << std::endl;
		return false;
	}
	for (const auto& imgPointsVec : imagePoints) {
		if (imgPointsVec.empty()) {
			std::cerr << "Error: Image points vectors must not be empty." << std::endl;
			return false;
		}
		for (const auto& pt : imgPointsVec) {
			if (std::isnan(pt.x) || std::isnan(pt.y)) {
				std::cerr << "Error: Image points must not contain NaNs." << std::endl;
				return false;
			}
		}
	}

	// 如果所有检查都通过，则返回 true
	return true;
}

struct CameraParameters {
	double rotation[3]; // 使用角轴表示的旋转。
	double translation[3]; // 平移。
	double* intrinsic; // 指向内参矩阵的指针。
	double* distortion; // 指向畸变系数的指针。
};


// 更新 monoCamera 对象的函数。
void UpdateMonoCamera(CameraController* camera, const CameraParameters& params, const double focus_factor) {
	cv::Mat rotation_vector;
	rotation_vector.create(3, 1, CV_64F);
	memcpy(rotation_vector.data, params.rotation, 3 * sizeof(double));
	cv::Rodrigues(rotation_vector, camera->R); // 将角轴旋转转换为旋转矩阵。
	camera->cameraMatrix.at<double>(1, 1) = camera->cameraMatrix.at<double>(0, 0) * focus_factor;
	camera->T.at<double>(0) = params.translation[0];
	camera->T.at<double>(1) = params.translation[1];
	camera->T.at<double>(2) = params.translation[2];
}

// 优化相机和点的函数。
void OptimizeCameraAndPoints(
	std::vector<CameraController*>& cameras,
	std::vector<cv::Point3d>& worldPoints,
	std::vector<std::vector<cv::Point2d>>& imagePoints) {

	double *focus_factor = new double[cameras.size()];

	CheckTypesForOptimization(cameras, worldPoints, imagePoints);
	ceres::Problem problem;
	std::vector<CameraParameters> camera_params(cameras.size()); // 存储所有相机的参数。

	for (size_t i = 0; i < cameras.size(); ++i) {
		// 初始化相机参数。
		camera_params[i].intrinsic = cameras[i]->cameraMatrix.ptr<double>();
		camera_params[i].distortion = cameras[i]->distCoeffs.ptr<double>();

		// 初始化焦距比值关系
		focus_factor[i] = camera_params[i].intrinsic[4] / camera_params[i].intrinsic[0];

		// 将旋转矩阵转换为角轴旋转
		cv::Rodrigues(cameras[i]->R, cv::Mat(3, 1, CV_64F, camera_params[i].rotation));
		memcpy(camera_params[i].translation, cameras[i]->T.ptr<double>(), 3 * sizeof(double));

		for (size_t j = 0; j < imagePoints[i].size(); ++j) {
			// 添加每个观测的残差块。
			ceres::CostFunction* cost_function =
				new ceres::AutoDiffCostFunction<ReprojectionErrorWithDistortion, 2, 3, 3, 3, 1, 1, 1, 1, 5>(
					new ReprojectionErrorWithDistortion(imagePoints[i][j]));
			problem.AddResidualBlock(cost_function, nullptr, camera_params[i].rotation, camera_params[i].translation,\
				&(worldPoints[j].x), &camera_params[i].intrinsic[0], \
				&camera_params[i].intrinsic[2], focus_factor + i, \
				&camera_params[i].intrinsic[5], camera_params[i].distortion);


			//problem.SetParameterBlockConstant(&(worldPoints[j].x));
			problem.SetParameterBlockConstant(&camera_params[i].intrinsic[0]);
			problem.SetParameterBlockConstant(focus_factor + i);
			problem.SetParameterBlockConstant(&camera_params[i].intrinsic[2]);
			problem.SetParameterBlockConstant(&camera_params[i].intrinsic[5]);
			problem.SetParameterBlockConstant(camera_params[i].distortion);
		}
	}

	// 配置和运行求解器。
	ceres::Solver::Options options;
	//options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	//options.trust_region_strategy_type = ceres::DOGLEG;
	options.minimizer_progress_to_stdout = false;
	options.max_num_iterations = 200;
	options.num_threads = 5;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	// 使用优化后的值更新相机参数。
	for (size_t i = 0; i < cameras.size(); ++i) {
		std::cout << *(focus_factor + i) << std::endl;
		UpdateMonoCamera(cameras[i], camera_params[i], *(focus_factor + i));
	}

	//std::cout << summary.FullReport() << std::endl;
}
