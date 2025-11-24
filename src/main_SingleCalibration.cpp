#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "argparse.h"

inline bool estimatePoseFromChessboard(const cv::Mat& image,
    const cv::Size& boardSize,
    double squareSize,
    const cv::Mat& K,
    const cv::Mat& distCoeffs,
    cv::Vec3d& rvec,
    cv::Vec3d& tvec,
    bool draw = false,
    cv::Mat* imageOut = nullptr,
    double* reprojErr = nullptr)
{
    if (image.empty() || boardSize.width < 2 || boardSize.height < 2 || squareSize <= 0.0) {
        std::cerr << "[estimatePoseFromChessboard] invalid input.\n";
        return false;
    }
    if (K.empty() || K.rows != 3 || K.cols != 3 || K.type() != CV_64F) {
        std::cerr << "[estimatePoseFromChessboard] camera matrix K must be 3x3 CV_64F.\n";
        return false;
    }
    if (distCoeffs.empty()) {
        std::cerr << "[estimatePoseFromChessboard] warning: distCoeffs is empty; assuming zero distortion.\n";
    }

    std::vector<cv::Point3d> objectPoints;
    objectPoints.reserve(static_cast<size_t>(boardSize.area()));
    for (int r = 0; r < boardSize.height; ++r) {
        for (int c = 0; c < boardSize.width; ++c) {
            objectPoints.emplace_back(c * squareSize, r * squareSize, 0.0);
        }
    }

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = image.clone();
    }

    std::vector<cv::Point2f> corners;
    const int findFlags =
        cv::CALIB_CB_ADAPTIVE_THRESH |
        cv::CALIB_CB_NORMALIZE_IMAGE;
    bool found = cv::findChessboardCorners(gray, boardSize, corners, findFlags);
    if (!found) {
        std::cerr << "[estimatePoseFromChessboard] chessboard not found.\n";
        return false;
    }

    cv::cornerSubPix(
        gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01)
    );

    int pnpFlag = 0;
#ifdef CV_SOLVEPNP_IPPE_SQUARE
    pnpFlag = cv::SOLVEPNP_IPPE_SQUARE;
#else
    pnpFlag = cv::SOLVEPNP_ITERATIVE;
#endif

    bool ok = cv::solvePnP(objectPoints, corners, K, distCoeffs, rvec, tvec, false, pnpFlag);
    if (!ok) {
        std::cerr << "[estimatePoseFromChessboard] solvePnP failed.\n";
        return false;
    }

    if (reprojErr) {
        std::vector<cv::Point2f> reprojected;
        cv::projectPoints(objectPoints, rvec, tvec, K, distCoeffs, reprojected);
        double errSum = 0.0;
        for (size_t i = 0; i < reprojected.size(); ++i) {
            errSum += cv::norm(reprojected[i] - corners[i]);
        }
        *reprojErr = errSum / static_cast<double>(reprojected.size());
    }

    if (draw) {
        cv::Mat vis;
        if (image.channels() == 1)
            cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
        else
            vis = image.clone();

        cv::drawChessboardCorners(vis, boardSize, corners, true);

        const float axisLen = static_cast<float>(3.0 * squareSize);
        std::vector<cv::Point3f> axisPts = {
            {0, 0, 0},
            {axisLen, 0, 0},
            {0, axisLen, 0},
            {0, 0, axisLen}
        };
        std::vector<cv::Point2f> axisImg;
        cv::projectPoints(axisPts, rvec, tvec, K, distCoeffs, axisImg);

        cv::line(vis, axisImg[0], axisImg[1], cv::Scalar(0, 0, 255), 2);   // X - ºì
        cv::line(vis, axisImg[0], axisImg[2], cv::Scalar(0, 255, 0), 2);   // Y - ÂÌ
        cv::line(vis, axisImg[0], axisImg[3], cv::Scalar(255, 0, 0), 2);   // Z - À¶

        if (imageOut) *imageOut = std::move(vis);
    }

    return true;
}

inline cv::Mat poseRtTo44(const cv::Vec3d& rvec, const cv::Vec3d& tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R); // 3x3, CV_64F
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    T.at<double>(0, 3) = tvec[0];
    T.at<double>(1, 3) = tvec[1];
    T.at<double>(2, 3) = tvec[2];
    return T;
}

std::string makeOutputPath(const std::string path) {
	std::filesystem::path p(path);
	auto dir = p.parent_path();
	auto outFile = dir / "output.txt";
	return outFile.string();
}

int NUM_HEIGHT = 0;
int NUM_WIDTH = 0;
float SQUARE_SIZE = 0.0;

cv::Mat intr = (cv::Mat_<double>(3, 3) <<
	1045.977, 0, 693.407,
	0, 1042.865, 581.199,
	0, 0, 1);
cv::Mat dist = (cv::Mat_<double>(1, 5) << -0.0751, -0.1446, 0, 0, 0.2794);

int main(int argc, char** argv) {
	argparse::ArgumentParser parser("Calibration");
	parser.add_argument("checkboard_image").help("image path");
	parser.add_argument("checkboard_num_height").help("Inner").scan<'i', int>();
	parser.add_argument("checkboard_num_width").help("Inner").scan<'i', int>();
	parser.add_argument("checkboard_size").help("in mm").scan<'g', float>();

	std::string checkboard_path;

	try {
		parser.parse_args(argc, argv);
		checkboard_path = parser.get<std::string>("checkboard_image");
		NUM_HEIGHT = parser.get<int>("checkboard_num_height");
		NUM_WIDTH = parser.get<int>("checkboard_num_width");
		SQUARE_SIZE = parser.get<float>("checkboard_size");
	}
	catch (const std::runtime_error& err) {
		std::cerr << err.what() << std::endl;
	}

	cv::Mat checkboard_img = cv::imread(checkboard_path, cv::IMREAD_COLOR);
	if (checkboard_img.empty()) {
		std::cerr << "Failed to load image: " << checkboard_img << std::endl;
		return -1;
	}

	cv::Vec3d rvec, tvec;
	cv::Mat image_out;
	double reproj_err = 0.0f;
	bool result = estimatePoseFromChessboard(
		checkboard_img,
		cv::Size(NUM_HEIGHT, NUM_WIDTH),
		SQUARE_SIZE,
		intr,
		dist,
		rvec,
		tvec,
		true,
		&image_out,
		nullptr);
	if (!result) {
		std::cerr << "Calibration failed" << std::endl;
		return -1;
	}

	cv::Mat pmat = poseRtTo44(rvec, tvec);
	std::cout << pmat << std::endl;

	std::ofstream outFile(makeOutputPath(checkboard_path));
	if (outFile.is_open()) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				outFile << " " + (i == 0 && j == 0) << pmat.at<double>(i, j);
			}
		}
		for (int i = 0; i < 3; ++i) {
			outFile << " " << pmat.at<double>(i, 3);
		}
		std::cout << std::endl;
		std::cout << "result written to: " << makeOutputPath(checkboard_path) << std::endl;
		outFile.close();
	}
	else {
		std::cerr << "Failed to open output file: " << makeOutputPath(checkboard_path) << std::endl;
	}

	std::cout << "reproj error: " << reproj_err << std::endl;
	cv::imshow("reproj", image_out);
	cv::waitKey(0);

	return 0;
}