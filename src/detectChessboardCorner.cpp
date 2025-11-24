#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include "detectChessboardCorner.h"
#include "constants.h"

/**
 * 提取图像中的角点
 * @param image 输入的图像
 * @param corners 角点的输出向量
 * @param maxCorners 最大角点数
 * @return 成功提取到角点返回true，否则返回false
 */
bool extractOneImageCorners(const cv::Mat& image,
    std::vector<cv::Point2f>& corners,
    int maxCorners /*= 100*/)
{
    cv::Mat grayImage;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    }
    else
    {
        grayImage = image.clone();
    }

    //pre process the image
    cv::Mat downsampleImage;
    downsampleImage = grayImage;
    //cv::resize(grayImage, downsampleImage, cv::Size(image.cols / 4, image.rows / 4), 0, 0, cv::INTER_LINEAR);
    // Apply bilateral filter to preserve edges while smoothing low frequency areas
    //cv::Mat filteredImage;
    //cv::bilateralFilter(downsampleImage, filteredImage, 9, 75, 75);
    //downsampleImage = filteredImage;

    bool patternWasFound = cv::findChessboardCorners(downsampleImage, cv::Size(NUM_HEIGHT, NUM_WIDTH), corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (patternWasFound)
    {
        // 恢复角点的原始尺寸
        //for (size_t j = 0; j < corners.size(); j++)
        //{
        //    corners[j].x *= 4;
        //    corners[j].y *= 4;
        //}
        //cv::cornerSubPix(grayImage, corners, cv::Size(3, 3), cv::Size(-1, -1),
        //    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }

    // findChessboard SB eq Matlab
    //bool patternWasFound = cv::findChessboardCornersSB(grayImage, cv::Size(NUM_HEIGHT, NUM_WIDTH), corners, cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_MARKER);
    drawChessboardCorners(grayImage, cv::Size(NUM_HEIGHT, NUM_WIDTH), corners, patternWasFound);

    return patternWasFound;
}

void rectifyCorner(std::vector<cv::Point2f>& corner2f) {
    cv::Size boardSize = cv::Size2i(NUM_HEIGHT, NUM_WIDTH);
    float squareSize = SQUARE_SIZE;

    // generate chessboard 3f points
    std::vector<cv::Point2f> objectPoints;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objectPoints.push_back(cv::Point2f(j * squareSize, i * squareSize));
        }
    }
    cv::Mat T = cv::findHomography(objectPoints, corner2f);
    perspectiveTransform(objectPoints, corner2f, T);

    //for (int i = 0; i < inputPts.rows; ++i) {
    //    float x = outputPts.at<float>(i, 0);
    //    float y = outputPts.at<float>(i, 1);
    //    corner2f.at(i) = cv::Point2f(x, y);
    //}
}

/**
 * 提取图像中的角点
 * @param imageList 输入的图像的列表
 * @param corners 角点的输出向量
 * @param maxCorners 最大角点数
 * @param cornersFound 被发现角点的索引
 * @return 成功提取到角点返回true，否则返回false
 */
bool extractMultiImageCorners(const std::vector<std::string>& imageList,
    std::vector<cv::Point2f>& corners,
    std::vector<bool>& cornersFound,
    int cornersCountPerImage = 88) 
{
    corners.clear();
    cornersFound.clear();
    for (size_t i = 0; i < imageList.size(); i++)
    {
        std::string fn = imageList[i];
        cv::Mat image = cv::imread(fn);

        if (image.empty())
        {
            std::cerr << "Cannot open image file: " << imageList[i] << std::endl;
            return false;
        }
        std::cout << "extract image file " << imageList[i] << std::endl;
        std::vector<cv::Point2f> corners2f;
        bool patternWasFound = extractOneImageCorners(image, corners2f, cornersCountPerImage);
        if (patternWasFound)
        {
            rectifyCorner(corners2f);
            cv::drawChessboardCorners(image, cv::Size(NUM_HEIGHT, NUM_WIDTH), corners2f, patternWasFound);

            //保存为新的文件
            std::string newFile = fn;
            size_t pos = newFile.find_last_of('.');
            if (pos != std::string::npos)
            {
                newFile = newFile.substr(0, pos) + "_corners" + newFile.substr(pos);
            }
            else
            {
                newFile += "_corners";
            }
            cv::imwrite(newFile, image);
        }
        else
        {
            std::cerr << "Cannot find chessboard corners in image file: " << fn << std::endl;
        }

        if (patternWasFound)
        {
            corners.insert(corners.end(), corners2f.begin(), corners2f.end());
        }
        cornersFound.push_back(patternWasFound);
        //saveOneImageFileCorners(fn, 88);
    }
    return true;
}

/**
 * 保存图像文件中的角点
 * @param outputFileName 输出的文件名
 * @param corners 角点的向量
 * @param cornersFound 被发现角点的索引
 * @return 成功保存角点返回true，否则返回false
 */
bool saveMultiImageFileCorners(std::string& outputFileName,
    std::vector<cv::Point2f>& corners,
    int cornersCountPerImage,
    std::vector<bool>& cornersFound)
{
    //保存为新的文件
    std::string newFile = outputFileName;
    std::ofstream ofs(newFile);
    if (!ofs.is_open())
    {
        std::cerr << "Cannot open file: " << newFile << std::endl;
        return false;
    }

    // write file header, the number of corners
    ofs << cornersFound.size() << std::endl;

    // write file header, the number of all corners
	ofs << corners.size()/cornersCountPerImage << std::endl;

    for (size_t i = 0; i < cornersFound.size(); i++)
    {
        ofs << cornersFound[i] << " ";
    }
    
    ofs << std::endl;

    for (size_t i = 0; i < corners.size(); i++)
    {
        ofs << corners[i].x << " " << corners[i].y << std::endl;
    }
    ofs.close();
    return true;
}

/**
 * 保存图像文件中的角点
 * @param imageFile 输入的图像文件
 * @param maxCorners 最大角点数
 * @return 成功保存角点返回true，否则返回false
 */
bool saveOneImageFileCorners(const std::string& imageFile, int maxCorners /*= 88*/)
{
    cv::Mat image = cv::imread(imageFile);
    if (image.empty())
    {
        std::cerr << "Cannot open image file: " << imageFile << std::endl;
        return false;
    }
    std::vector<cv::Point2f> corners;
    bool patternWasFound = extractOneImageCorners(image, corners, maxCorners);
    if (patternWasFound)
    {
        cv::drawChessboardCorners(image, cv::Size(NUM_HEIGHT, NUM_WIDTH), corners, patternWasFound);

        //保存为新的文件
        std::string newFile = imageFile;
        size_t pos = newFile.find_last_of('.');
        if (pos != std::string::npos)
        {
            newFile = newFile.substr(0, pos) + "_corners" + newFile.substr(pos);
        }
        else
        {
            newFile += "_corners";
        }
        cv::imwrite(newFile, image);
    }
    else
    {
        std::cerr << "Cannot find chessboard corners in image file: " << imageFile << std::endl;
    }
    return patternWasFound;
}

bool detectOneImageFileCorners(const std::string& imageFile, std::vector<double>& corners, int maxCorners /*= 88*/)
{
    cv::Mat image = cv::imread(imageFile);
    if (image.empty())
    {
        std::cerr << "Cannot open image file: " << imageFile << std::endl;
        return false;
    }
    std::vector<cv::Point2f> corners2f;
    bool patternWasFound = extractOneImageCorners(image, corners2f, maxCorners);
    if (patternWasFound)
    {
        corners.assign(corners2f.size() * 2, 0.0);
        for (size_t i = 0; i < corners2f.size(); i++)
        {
            corners[i * 2] = corners2f[i].x;
            corners[i * 2 + 1] = corners2f[i].y;
        }
    }
    else
    {
        std::cerr << "Cannot find chessboard corners in image file: " << imageFile << std::endl;
    }
    return patternWasFound;
}

bool readAllImageFiles(const std::string& imagedir, std::vector<std::string>& imageList)
{
    imageList.clear();
    std::string suffix = ".bmp";
    std::string pattern = imagedir + "/*" + suffix;
    std::vector<cv::String> fn;
    cv::glob(pattern, fn, false);
    for (size_t i = 0; i < fn.size(); i++)
    {
        std::string filename = fn[i];
        // 找到最后一个'/'的位置，以便提取出文件名部分（不包含路径）
        size_t lastSlashIndex = filename.find_last_of('/');
        if (lastSlashIndex == std::string::npos)
        {
            lastSlashIndex = filename.find_last_of('\\');
        }
        std::string pureFilename = filename.substr(lastSlashIndex + 1);

        // 检查文件名中是否包含'_'，如果不包含则添加到imageList中
        if (pureFilename.find('_') == std::string::npos)
        {
            imageList.push_back(fn[i]);
        }
    }
    return !imageList.empty();
}


bool detectMultiImageFileCorners(const std::string& imagedir, int cornersCountPerImage)
{
    std::vector<std::string> imageList;
    if (!readAllImageFiles(imagedir, imageList))
    {
        std::cerr << "Cannot read image files from directory: " << imagedir << std::endl;
        return false;
    }
    std::vector<cv::Point2f> corners;
    std::vector<bool> cornersFound;
    if (!extractMultiImageCorners(imageList, corners, cornersFound, cornersCountPerImage))
    {
        return false;
    }
    std::string outputFileName = imagedir + "/corners.txt";
    if (!saveMultiImageFileCorners(outputFileName, 
        corners, cornersCountPerImage , cornersFound))
    {
        return false;
    }
    return true;
}




