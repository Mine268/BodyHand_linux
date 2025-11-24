#define _CRT_SECURE_NO_WARNINGS

#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "argparse.h"
#include "Pose3D.h"
#include "CMultiCap.h"
#include "tcp_server.h"

const std::vector<std::pair<int, int>> BODY_CONNECTION{
	{0, 1}, {0, 2}, {1, 3}, {2, 4},
	{5, 6}, {6, 12}, {12, 11}, {11, 5},
	{5, 7}, {7, 9}, {6, 8}, {8, 10},
	{11, 13}, {13, 15}, {12, 14}, {14, 16}
};
const std::vector<std::pair<int, int>> HAND_CONNECTION{
	{0, 1}, {1, 2}, {2, 3}, {3, 4},
	{5, 9}, {9, 13}, {13, 17}, {17, 0}, {0, 5},
	{5, 6}, {6, 7}, {7, 8},
	{9, 10}, {10, 11}, {11, 12},
	{13, 14}, {14, 15}, {15, 16},
	{17, 18}, {18, 19}, {19, 20}
};

std::string get_time_string() {
	// 1. 获取当前的高精度时间点
	auto now = std::chrono::system_clock::now();

	// 2. 将时间点转换为 std::time_t (秒级精度) 和秒级时间点
	auto now_seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
	std::time_t now_c = std::chrono::system_clock::to_time_t(now_seconds);

	// 3. 计算自秒初以来的毫秒数
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - now_seconds);
	int milliseconds = static_cast<int>(ms.count());

	// 4. 将 std::time_t 转换为本地时间结构体
	// 注意：std::localtime 不是线程安全的，在多线程环境应使用 std::localtime_r (POSIX) 或其他线程安全方法
	std::tm* parts = std::localtime(&now_c);

	// 5. 格式化日期和时间 (不含毫秒)
	// 使用 std::put_time 和 std::ostringstream 避免固定大小的缓冲区问题，并与 C++ I/O 更好地集成
	std::ostringstream oss;

	// 格式化为 YYYYMMDD_HHMMSS
	// 注意：在某些旧的或非标准库实现中，std::put_time 可能不可用，
	// 此时你可以退回到使用 std::strftime 和 char 缓冲区
	oss << std::put_time(parts, "%Y%m%d_%H%M%S");

	// 6. 添加毫秒数，确保有三位数，不足补零
	oss << "_" << std::setfill('0') << std::setw(3) << milliseconds;

	return oss.str();
}

std::string get_tcp_data_string(const BodyHand::PoseResult& pose_result) {
	std::string data{};
	// Body
	if (pose_result.valid_body) {
		for (const auto& kp : pose_result.body_kps_3d) {
			data += std::to_string(kp.x) + ", " + std::to_string(kp.y) + ", " + std::to_string(kp.z) + ", ";
		}
	}
	else {
		for (size_t i = 0; i < 17; ++i) {
			data += "0, 0, 0, ";
		}
	}
	// Hand
	if (pose_result.valid_left && pose_result.valid_right) {
		for (const auto& kp : pose_result.hand_kps_3d) {
			data += std::to_string(kp.x) + ", " + std::to_string(kp.y) + ", " + std::to_string(kp.z) + ", ";
		}
	}
	else {
		for (size_t i = 0; i < 42; ++i) {
			data += "0, 0, 0, ";
		}
	}
	// 移除最后的逗号和空格
	if (data.size() >= 2) {
		data.pop_back();
		data.pop_back();
	}
	return data;
}

BodyHand::PoseResult apply_trans(
	const BodyHand::PoseResult& pose,
	const cv::Mat &rmat /* (3,3) */,
	const cv::Mat &tvec /* (3,1) */) {
	BodyHand::PoseResult new_pose = pose;
	if (pose.valid_body) {
		for (auto& kp : new_pose.body_kps_3d) {
			cv::Mat pt = (cv::Mat_<float>(3, 1) << kp.x, kp.y, kp.z);
			cv::Mat new_pt = rmat * pt + tvec;
			kp.x = new_pt.at<float>(0);
			kp.y = new_pt.at<float>(1);
			kp.z = new_pt.at<float>(2);
		}
	}
	if (pose.valid_left) {
		for (size_t i = 0; i < 21; ++i) {
			cv::Point3f& kp = new_pose.hand_kps_3d[i];
			cv::Mat pt = (cv::Mat_<float>(3, 1) << kp.x, kp.y, kp.z);
			cv::Mat new_pt = rmat * pt + tvec;
			kp.x = new_pt.at<float>(0);
			kp.y = new_pt.at<float>(1);
			kp.z = new_pt.at<float>(2);
		}
	}
	if (pose.valid_right) {
		for (size_t i = 21; i < 42; ++i) {
			cv::Point3f& kp = new_pose.hand_kps_3d[i];
			cv::Mat pt = (cv::Mat_<float>(3, 1) << kp.x, kp.y, kp.z);
			cv::Mat new_pt = rmat * pt + tvec;
			kp.x = new_pt.at<float>(0);
			kp.y = new_pt.at<float>(1);
			kp.z = new_pt.at<float>(2);
		}
	}
	return new_pose;
}

cv::Mat plot_2d_result(const std::vector<cv::Mat>& imgs, const BodyHand::PoseResult& pose, int idx) {
	cv::Mat img_vis = imgs[idx].clone();
	if (pose.valid_body) {
		for (size_t i = 0; i < pose.body_kps_2d[idx].size(); ++i) {
			const auto& kps = pose.body_kps_2d[idx][i];
			for (size_t j = 0; j < kps.size(); ++j) {
				cv::circle(img_vis, kps[j], 3, cv::Scalar(0, 255, 0), -1);
			}
			for (const auto& conn : BODY_CONNECTION) {
				if (kps[conn.first].x > 0 && kps[conn.first].y > 0 &&
					kps[conn.second].x > 0 && kps[conn.second].y > 0) {
					cv::line(img_vis, kps[conn.first], kps[conn.second], cv::Scalar(255, 0, 0), 2);
				}
			}
		}
	}
	if (pose.valid_left) {
		const auto& kps = pose.hand_kps_2d;
		for (size_t j = 0; j < 21; ++j) {
			cv::circle(img_vis, kps[j], 3, cv::Scalar(0, 255, 255), -1);
		}
		for (const auto& conn : HAND_CONNECTION) {
			if (kps[conn.first].x > 0 && kps[conn.first].y > 0 &&
				kps[conn.second].x > 0 && kps[conn.second].y > 0) {
				cv::line(img_vis, kps[conn.first], kps[conn.second], cv::Scalar(255, 255, 0), 2);
			}
		}
	}
	if (pose.valid_right) {
		const auto& kps = pose.hand_kps_2d;
		for (size_t j = 21; j < 42; ++j) {
			cv::circle(img_vis, kps[j], 3, cv::Scalar(0, 255, 255), -1);
		}
		for (const auto& conn : HAND_CONNECTION) {
			if (kps[conn.first + 21].x > 0 && kps[conn.first + 21].y > 0 &&
				kps[conn.second + 21].x > 0 && kps[conn.second + 21].y > 0) {
				cv::line(img_vis, kps[conn.first + 21], kps[conn.second + 21], cv::Scalar(255, 255, 0), 2);
			}
		}
	}
	return img_vis;
}

cv::Mat plot_3d_reproj_result(const cv::Mat& img, const BodyHand::PoseResult& pose, const cv::Mat& intr) {
	cv::Mat img_vis = img.clone();

	if (pose.valid_body) {
		cv::Mat kps3d_mat(pose.body_kps_3d.size(), 3, CV_32F);
		for (size_t j = 0; j < pose.body_kps_3d.size(); ++j) {
			kps3d_mat.at<float>(j, 0) = pose.body_kps_3d[j].x; // X
			kps3d_mat.at<float>(j, 1) = pose.body_kps_3d[j].y; // Y
			kps3d_mat.at<float>(j, 2) = pose.body_kps_3d[j].z; // Z
		}

		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_32F); // 旋转向量 (0, 0, 0)
		cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F); // 平移向量 (0, 0, 0)
		cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32F); // 假设无畸变或使用零矩阵占位

		std::vector<cv::Point2f> projected_points;
		cv::projectPoints(kps3d_mat, rvec, tvec, intr, distCoeffs, projected_points);
		for (const auto& pt2d : projected_points) {
			// 检查点是否在图像范围内，避免绘图错误
			if (pt2d.x >= 0 && pt2d.x < img_vis.cols &&
				pt2d.y >= 0 && pt2d.y < img_vis.rows) {
				// 绘制关键点（例如，一个红色小圆圈）
				cv::circle(img_vis, pt2d, 3, cv::Scalar(0, 255, 0), -1 );
			}
		}
		for (const auto& conn : BODY_CONNECTION) {
			if (projected_points[conn.first].x > 0 && projected_points[conn.first].y > 0 &&
				projected_points[conn.second].x > 0 && projected_points[conn.second].y > 0) {
				cv::line(img_vis, projected_points[conn.first], projected_points[conn.second], cv::Scalar(255, 0, 0), 2);
			}
		}
	}

	if (pose.valid_left) {
		const int LEFT_HAND_KPS_COUNT = 21;
		cv::Mat kps3d_hand_mat(LEFT_HAND_KPS_COUNT, 3, CV_32F);
		for (size_t j = 0; j < LEFT_HAND_KPS_COUNT; ++j) {
			if (j < pose.hand_kps_3d.size()) {
				kps3d_hand_mat.at<float>(j, 0) = pose.hand_kps_3d[j].x; // X
				kps3d_hand_mat.at<float>(j, 1) = pose.hand_kps_3d[j].y; // Y
				kps3d_hand_mat.at<float>(j, 2) = pose.hand_kps_3d[j].z; // Z
			}
		}

		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_32F);
		cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F);
		cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32F);

		std::vector<cv::Point2f> projected_hand_points;
		cv::projectPoints(kps3d_hand_mat, rvec, tvec, intr, distCoeffs, projected_hand_points);
		for (const auto& pt2d : projected_hand_points) {
			// 检查点是否在图像范围内
			if (pt2d.x >= 0 && pt2d.x < img_vis.cols &&
				pt2d.y >= 0 && pt2d.y < img_vis.rows) {
				// 绘制左手关键点（黄色，与 2D 绘图颜色一致）
				cv::circle(img_vis, pt2d, 3, cv::Scalar(0, 255, 255), -1);
			}
		}
		for (const auto& conn : HAND_CONNECTION) {
			if (projected_hand_points[conn.first].x > 0 && projected_hand_points[conn.first].y > 0 &&
				projected_hand_points[conn.second].x > 0 && projected_hand_points[conn.second].y > 0) {
				cv::line(img_vis, projected_hand_points[conn.first], projected_hand_points[conn.second], cv::Scalar(255, 255, 0), 2);
			}
		}
	}

	if (pose.valid_right) {
		const size_t HAND_KPS_COUNT = 21;
		const size_t RIGHT_HAND_START_IDX = HAND_KPS_COUNT; // 21

		cv::Mat kps3d_hand_mat(HAND_KPS_COUNT, 3, CV_32F);
		for (size_t j = 0; j < HAND_KPS_COUNT; ++j) {
			size_t global_idx = RIGHT_HAND_START_IDX + j;
			if (global_idx < pose.hand_kps_3d.size()) {
				kps3d_hand_mat.at<float>(j, 0) = pose.hand_kps_3d[global_idx].x;
				kps3d_hand_mat.at<float>(j, 1) = pose.hand_kps_3d[global_idx].y;
				kps3d_hand_mat.at<float>(j, 2) = pose.hand_kps_3d[global_idx].z;
			}
		}

		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_32F);
		cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F);
		cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32F);

		std::vector<cv::Point2f> projected_hand_points;
		cv::projectPoints(kps3d_hand_mat, rvec, tvec, intr, distCoeffs, projected_hand_points);
		for (const auto& pt2d : projected_hand_points) {
			if (pt2d.x >= 0 && pt2d.x < img_vis.cols && pt2d.y >= 0 && pt2d.y < img_vis.rows) {
				cv::circle(img_vis, pt2d, 3, cv::Scalar(0, 255, 255), -1); // 手部点 (黄色)
			}
		}
		for (const auto& conn : HAND_CONNECTION) {
			if (projected_hand_points[conn.first].x > 0 && projected_hand_points[conn.first].y > 0 &&
				projected_hand_points[conn.second].x > 0 && projected_hand_points[conn.second].y > 0) {
				cv::line(img_vis, projected_hand_points[conn.first], projected_hand_points[conn.second], cv::Scalar(255, 255, 0), 2);
			}
		}
	}

	return img_vis;
}

int main(int argc, char** argv) {

	// ******** 参数解析 ********
	std::string config_path;
	std::vector<int> tcp_ports;
	bool send_tcp = false;
	bool no_write_file = false;

	argparse::ArgumentParser parser("Pose estimation");
	parser.add_description(
		"从配置文件中读取模型信息和相机标定信息，进行姿态估计。"
		"\n\t第一、二、三行：分别是人体姿态估计模型，人体检测模型，手部姿态估计模型的地址"
		"\n\t第四行是一个正整数n，表示总共有多少个视图"
		"\n\t接下来的n行每行有26个浮点数，前9个表示内参矩阵，接着的9个表示旋转变换矩阵，然后3个表示位移变换向量，最后的5个表示畸变参数"
		"\n\t最后一行是可写可不写的，用来表示如何从相机坐标系变换到全局坐标系，包含12个浮点数，描述的是全局坐标系在相机坐标系下的三轴方向和原点偏移"
	);
	parser.add_argument("config_path").help("配置文件地址");
	//parser.add_argument("--send_tcp").help("是否通过TCP发送姿态数据").default_value(false).implicit_value(true).nargs(0);
	parser.add_argument("--send_tcp").help("要进行数据发送的一个或多个端口").scan<'i', int>().nargs(argparse::nargs_pattern::at_least_one);
	parser.add_argument("--no_write_file").help("是否将结果写入文件").default_value(false).implicit_value(true).nargs(0);
	try {
		parser.parse_args(argc, argv);
		config_path = parser.get<std::string>("config_path");
		//send_tcp = parser.get<bool>("--send_tcp");
		if (parser.is_used("--send_tcp")) {
			send_tcp = true;
			tcp_ports = parser.get<std::vector<int>>("--send_tcp");
		}
		no_write_file = parser.get<bool>("--no_write_file");
	}
	catch (const std::runtime_error& err) {
		std::cerr << err.what() << std::endl;
	}

	// ******** 从文件读取初始化配置 ********
	BodyHand::BodyModelConfig body_cfg;
	BodyHand::HandModelConfig hand_cfg;
	int num_view;
	// 双视图标定
	std::vector<cv::Mat> intr;
	std::vector<cv::Mat> rot_trans;
	std::vector<cv::Mat> transl_trans;
	std::vector<std::vector<float>> undist;
	// 单视图标定
	cv::Mat rot_world, trans_world;

	std::ifstream config_file(config_path);
	// 配置文件读取
	if (!config_file.is_open()) {
		std::cerr << "无法打开配置文件：" << config_path << std::endl;
		config_file.close();
		return -1;
	}
	// 模型文件地址读取
	if (!std::getline(config_file, body_cfg.model_path)) {
		std::cerr << "无法读取人体姿态估计模型件地址。" << std::endl;
		config_file.close();
		return -1;
	}
	if (!std::getline(config_file, hand_cfg.handlr_path)) {
		std::cerr << "无法读取人体检测模型文件地址。" << std::endl;
		config_file.close();
		return -1;
	}
	if (!std::getline(config_file, hand_cfg.hamer_path)) {
		std::cerr << "无法读取手部姿态估计模型文件地址。" << std::endl;
		config_file.close();
		return -1;
	}
	// 相机信息读取
	if (!(config_file >> num_view)) {
		std::cerr << "无法读取视图数量。" << std::endl;
		config_file.close();
		return -1;
	}
	// 逐行读取数据
	for (int i = 0; i < num_view; ++i) {
		// 临时数组存储一行26个float
		float data[26];
		for (int j = 0; j < 26; ++j) {
			if (!(config_file >> data[j])) { // 读取失败处理
				std::cerr << "文件格式错误或数据不足。" << std::endl;
				config_file.close();
				return -1;
			}
		}
		// 1. 内参矩阵 3x3 (前9个元素)
		intr.emplace_back(cv::Mat(3, 3, CV_32F, data).clone());
		// 2. 旋转矩阵 3x3 (中间9个元素)
		rot_trans.emplace_back(cv::Mat(3, 3, CV_32F, data + 9).clone());
		// 3. 平移向量 3x1 (接下来3个元素)
		transl_trans.emplace_back(cv::Mat(3, 1, CV_32F, data + 18).clone());
		// 4. 畸变参数 5个 (最后5个元素)
		undist.emplace_back(std::vector<float> {data[21], data[22], data[23], data[24], data[25]});
	}
	// 最后12个参数是单视图标定数据，如果没有提供则使用单位阵和零向量
	bool world_coord_data_valid{ true };
	float world_coord_data[12];
	for (int j = 0; j < 12; ++j) {
		if (!(config_file >> world_coord_data[j])) { // 读取失败处理
			std::cerr << "无法读取单视图标定数据。" << std::endl;
			world_coord_data_valid = false;
			break;
		}
	}
	// 读取单视图标定数据
	if (world_coord_data_valid) {
		rot_world = cv::Mat(3, 3, CV_32F, world_coord_data);
		trans_world = cv::Mat(3, 1, CV_32F, world_coord_data + 9);
		// 求逆
		rot_world = rot_world.t();
		trans_world = -rot_world * trans_world;
	}
	else { // 如果没有提供单视图标定数据，则使用单位阵和零向量，即姿态位于相机坐标系
		rot_world = cv::Mat3f::eye(3, 3);
		trans_world = cv::Mat3f::ones(3, 1);
	}
	std::cout << "配置文件读取完成。" << std::endl;
	config_file.close();

	// ******** 构造姿态估计器 ********
	BodyHand::PoseEstimator pe{
		body_cfg,
		hand_cfg,
		num_view,
		intr,
		rot_trans,
		transl_trans,
		undist
	};
	std::cout << "姿态估计器构造完成。" << std::endl;

	// ******** TCP服务端 ********
	std::vector<SOCKET> listen_sockets;
	std::vector<SOCKET> client_sockets;
	if (!tcp_ports.empty()) {
		if (!InitSockets()) {
			std::cerr << "Winsock 初始化失败，无法启动 TCP 服务器。" << std::endl;
			return -1;
		}

		std::cout << "正在初始化 " << tcp_ports.size() << " 个 TCP 端口..." << std::endl;

		for (int port : tcp_ports) {
			std::string port_str = std::to_string(port);

			// 创建监听 Socket
			SOCKET ls = CreateListeningSocket(port_str.c_str());
			if (ls == INVALID_SOCKET) {
				std::cerr << "[Port " << port << "] 创建监听失败，跳过此端口。" << std::endl;
				continue;
			}
			listen_sockets.push_back(ls);

			std::cout << "[Port " << port << "] 等待客户端连接..." << std::endl;

			// 阻塞等待客户端连接
			// 注意：这里是串行阻塞。必须等到 Port A 连上，才会去监听 Port B
			SOCKET cs = AcceptClient(ls);
			if (cs == INVALID_SOCKET) {
				std::cerr << "[Port " << port << "] 接受连接失败。" << std::endl;
				CloseSocket(ls); // 如果连接失败，关闭对应的监听
				// 即使失败也继续尝试下一个端口，还是直接退出取决于你的需求
				// 这里选择继续，但该端口将不可用
			}
			else {
				client_sockets.push_back(cs);
				std::cout << "[Port " << port << "] 客户端已连接！" << std::endl;
			}
		}
		if (client_sockets.empty()) {
			std::cerr << "错误：没有任何客户端成功连接，程序退出。" << std::endl;
			CleanupSockets();
			return -1;
		}
		std::cout << "所有端口准备就绪，共 " << client_sockets.size() << " 个客户端。" << std::endl;
	}
	//SOCKET ls{}, cs{};
	//if (send_tcp) {
	//	if (!InitWinsock()) {
	//		std::cerr << "Winsock 初始化失败，无法启动 TCP 服务器。" << std::endl;
	//		return -1;
	//	}
	//	ls = CreateListeningSocket("5175");
	//	if (ls == INVALID_SOCKET) {
	//		std::cerr << "创建监听套接字失败，无法启动 TCP 服务器。" << std::endl;
	//		WSACleanup();
	//		return -1;
	//	}
	//	cs = AcceptClient(ls);
	//	if (cs == INVALID_SOCKET) {
	//		std::cerr << "接受客户端连接失败，无法启动 TCP 服务器。" << std::endl;
	//		CloseSocket(ls);
	//		CleanupWinsock();
	//		return -1;
	//	}
	//	std::cout << "客户端已连接，可以发送数据。" << std::endl;
	//}

	// ******** 构造捕捉系统 ********
	get_app();
	init_device();
	if (num_view != get_device_count()) {
		std::cerr << "配置文件的视图数目为：" << num_view << "，而接入的相机数目为：" << get_device_count() << std::endl;
		close_device();
		return -1;
	}
	start_grabbing();

	// ******** 捕捉和姿态估计 ********
	std::filesystem::path config_path_obj(config_path);
	std::string pose_output_path = config_path_obj.parent_path().string() + '/' + get_time_string() + "_pose_output.txt";
	std::ofstream pose_ofs; // 按照当前日期命名的姿态输出文件
	if (!no_write_file) {
		pose_ofs.open(pose_output_path);
		std::cout << "姿态数据保存至：" << pose_output_path << std::endl;
	}
	else {
		std::cout << "不保存姿态数据。" << std::endl;
	}
	do {
		auto cap_info = capture();
		if (cap_info.flag) {
			std::vector<cv::Mat> imgs;
			BodyHand::PoseResult pose_result;
			for (int i = 0; i < num_view; ++i) {
				cv::Mat img_rgb(cap_info.height[i], cap_info.width[i], CV_8UC3, cap_info.ppbuffer[i]), img_bgr;
				cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);
				//cv::imshow(std::format("{}", i), img_bgr);
				imgs.emplace_back(std::move(img_bgr));
			}
			if (cv::waitKey(20) == 'q') {
				break;
			}

			// 进行姿态估计
			pe.estimatePose(imgs, pose_result, 0);
			if (pose_result.valid_body && pose_result.valid_left && pose_result.valid_right) {
				cv::Mat img_3d_reproj = plot_3d_reproj_result(imgs[0], pose_result, intr[0]);
				cv::imshow("3D reproj", img_3d_reproj);
			}
			// 将相机空间姿态变换到全局坐标系
			pose_result = apply_trans(pose_result, rot_world, trans_world);

			if (pose_result.valid_body && pose_result.valid_left && pose_result.valid_right) {
				std::cout << "Found pose\n";
				std::string time_string = get_time_string();

				// 可视化手部检测
				//cv::Mat hand_ref_img = imgs[0];
				//for (const auto& rect : pose_result.hand_bbox) {
				//	cv::rectangle(hand_ref_img, rect, cv::Scalar(0, 255, 0), 2);
				//}
				//cv::imshow("Hand detection", hand_ref_img);

				// 可视化2D结果
				//cv::Mat img_2d = plot_2d_result(imgs, pose_result, 0);
				//cv::imshow("2D Result", img_2d);

				if (!no_write_file) {
					// 存储到文件，每行第一个是时间戳
					pose_ofs << time_string << ' ';
					// 然后是全部3d人体关节点坐标
					for (const auto& kp : pose_result.body_kps_3d) {
						pose_ofs << kp.x << ' ' << kp.y << ' ' << kp.z << ' ';
					}
					// 然后是全部42个手部关节点坐标，先左后右
					for (const auto& kp : pose_result.hand_kps_3d) {
						pose_ofs << kp.x << ' ' << kp.y << ' ' << kp.z << ' ';
					}
					pose_ofs << '\n';
				}

				// 如果开启了TCP发送，则拼接字符串并发送：x1, y1, z1, x2, y2, z2, ...
				if (!send_tcp) continue;
				//std::string tcp_data = get_tcp_data_string(pose_result);
				//SendTextLine(cs, tcp_data);
				if (!client_sockets.empty()) {
					std::string tcp_data = get_tcp_data_string(pose_result);
					// 遍历发送
					for (size_t i = 0; i < client_sockets.size(); ++i) {
						// 注意：如果某个客户端断开，SendTextLine 会失败
						// 这里简单的实现是尝试发送，如果失败打印错误但不崩溃
						bool success = SendTextLine(client_sockets[i], tcp_data);
						if (!success) {
							// 实际项目中可能需要在这里移除断开的 socket，
							// 但为了代码简单，这里仅打印错误。
							// WSAGetLastError() 可以查看具体错误
							std::cerr << "向客户端 " << i << " 发送失败" << std::endl;
						}
					}
				}
			}
		}
	} while (true);
	if (!no_write_file) {
		pose_ofs.close();
	}

	// ******** 关闭TCP连接 ********
	if (!tcp_ports.empty()) {
		std::cout << "关闭 TCP 服务器..." << std::endl;
		for (SOCKET s : client_sockets) CloseSocket(s);
		for (SOCKET s : listen_sockets) CloseSocket(s);
		CleanupSockets();
	}

	// ******** 停止捕捉和释放资源 ********
	stop_grabbing();
	close_device();

	return 0;
}