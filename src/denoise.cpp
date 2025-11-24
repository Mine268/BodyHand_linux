#include "denoise.h"
#include <opencv2/opencv.hpp>
#include <opencv2/photo/cuda.hpp>

bool NIMeansDenoisingColor(const std::string srcpath, const std::string outpath)
{
	cv::Mat image = cv::imread(srcpath);
	cv::Mat denoiseImage;
	cv::fastNlMeansDenoisingColored(image, denoiseImage);
	cv::imwrite(outpath, denoiseImage);
	return true;
}

bool NIMeansDenoising(const std::string srcpath, const std::string outpath)
{
	cv::Mat image = cv::imread(srcpath);
	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
	cv::Mat denoiseImage;
	cv::fastNlMeansDenoising(grayImage, denoiseImage);
	cv::imwrite(outpath, denoiseImage);
	return true;
}

bool nonLocalMeans(const std::string srcpath, const std::string outpath)
{
	cv::Mat image = cv::imread(srcpath);
	cv::cuda::GpuMat image_cuda;
	cv::Mat denoiseImage;
	cv::cuda::GpuMat denoiseImage_cuda;
	image_cuda.upload(image);
	denoiseImage_cuda.upload(denoiseImage);

	cv::cuda::nonLocalMeans(image_cuda, denoiseImage_cuda,5);

	denoiseImage_cuda.download(denoiseImage);
	cv::imwrite(outpath, denoiseImage);
	return true;
}
