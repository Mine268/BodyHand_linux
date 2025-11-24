#pragma once
#include <string>

bool NIMeansDenoisingColor(const std::string srcpath, const std::string outpath);

bool NIMeansDenoising(const std::string srcpath, const std::string outpath);

bool nonLocalMeans(const std::string srcpath, const std::string outpath);
