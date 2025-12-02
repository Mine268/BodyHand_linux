#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <map>
#include <vector>
#include <string>
#include "YoloOnnxDet.h"   // 复用 Detection 结构体

namespace BodyHand {

    class VitInference {
    public:
        // pose_model: ViTPose onnx 路径
        // pose_input_size: ViTPose 输入尺寸，例如 (256,192) / (384,288)
        // device: 这里主要控制是否用 CUDA EP，可在 cpp 里调整
        VitInference(const std::string& pose_model,
                    const cv::Size& pose_input_size = cv::Size(256, 192),
                    const std::string& device = "cuda");

        // img_rgb: RGB 图
        // dets: YOLO 检测框
        // 返回: id -> (num_joints x 3) 的 Mat, 每行 (x, y, score)
        std::map<int, cv::Mat> inference(const cv::Mat& img_rgb,
                                        const std::vector<Detection>& dets);

        // 在 RGB 图上画框+骨架，并返回 RGB 图
        cv::Mat draw(const cv::Mat& img_rgb,
                    const std::map<int, cv::Mat>& kpts_map,
                    const std::vector<Detection>& dets,
                    float kp_conf_thr = 0.5f,
                    bool show_yolo = true);

    private:
        cv::Size target_size_;
        int      num_joints_;

        Ort::Env            env_;
        Ort::SessionOptions sess_opts_;
        Ort::Session        sess_;
        std::string         input_name_;
        std::string         output_name_;

        std::vector<float> mean_;
        std::vector<float> std_;

        // 简单 COCO skeleton
        std::vector<std::pair<int,int>> skeleton_;

        cv::Mat padImage(const cv::Mat& img, float aspect, int& top, int& left);
        cv::Mat preImg(const cv::Mat& img, int& org_h, int& org_w);
        cv::Mat postprocess(const std::vector<float>& heatmaps,
                            int c, int h, int w,
                            int org_w, int org_h);
    };

}