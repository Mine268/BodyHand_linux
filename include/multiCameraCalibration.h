#pragma once
#include <string>

/**
 * 多视图相机标定
 */

bool multiCameraCalibrationAndSave(std::string& calibrationDir,
	std::string& viewName, int cornersCountPerImage, bool is_fix_instrinic);
