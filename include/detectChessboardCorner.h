#pragma once
#include <vector>
#include <string>


/**
 * 提取图像中的角点
 * @param imageFile 输入的图像途径
 * @param maxCorners 最大角点数
 * @return 成功提取到角点返回true，否则返回false
 */
bool saveOneImageFileCorners(const std::string& imageFile,
    int maxCorners = 88);


/**
 * 提取目录中图像的角点，并保存到文件中
 * @param imagedir 输入的图像目录
 * @return 成功提取到角点返回true，否则返回false
 */
bool detectMultiImageFileCorners(const std::string& imagedir, int cornersCountPerImage);
