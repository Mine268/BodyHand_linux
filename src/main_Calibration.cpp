#include <filesystem>
#include <iostream>
#include <exception>
#include "argparse.h"
#include "multiCameraCalibration.h"
#include "visualization.h"
#include "constants.h"

int NUM_HEIGHT = 0;
int NUM_WIDTH = 0;
float SQUARE_SIZE = 0.0;

int main(int argc, char **argv) {
	argparse::ArgumentParser parser("Calibration");
	parser.add_argument("root").help("Path to multiview images root");
	parser.add_argument("prefix").help("Prefix of each image folder");
	parser.add_argument("checkboard_num_height").help("Inner").scan<'i', int>();
	parser.add_argument("checkboard_num_width").help("Inner").scan<'i', int>();
	parser.add_argument("checkboard_size").help("in mm").scan<'g', float>();

	std::string root;
	std::string prefix;

	try {
		parser.parse_args(argc, argv);
		root = parser.get<std::string>("root");
		prefix = parser.get<std::string>("prefix");
		NUM_HEIGHT = parser.get<int>("checkboard_num_height");
		NUM_WIDTH = parser.get<int>("checkboard_num_width");
		SQUARE_SIZE = parser.get<float>("checkboard_size");
	}
	catch (const std::runtime_error& err) {
		std::cerr << err.what() << std::endl;
	}

	std::string calibrationRoot = root;
	std::string viewName = prefix;
	bool result = multiCameraCalibrationAndSave(calibrationRoot, viewName, 54, true);
	if (!result) {
		std::cerr << "Calibration failed" << std::endl;
	}
	return 0;
}