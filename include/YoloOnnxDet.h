#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

namespace BodyHand {

    // 与前面 VitInference 中的 Detection 兼容
    struct Detection {
        cv::Rect bbox;   // x, y, w, h
        float score;
        int id;          // 这里检测阶段可设为 -1
        int cls;         // 类别索引
    };

    class YoloOnnx {
    public:
        YoloOnnx(const std::string& onnx_path,
                const cv::Size& input_size = cv::Size(640, 640),
                float conf_thres = 0.25f,
                float iou_thres  = 0.45f);

        // img_bgr：OpenCV 读进来的 BGR 图
        // allowed_cls：如果为空，则不过滤类别；否则只保留列表里的类别
        std::vector<Detection> infer(const cv::Mat& img_bgr,
                                    const std::vector<int>& allowed_cls = {});

    private:
        cv::Size input_size_;
        float conf_thres_;
        float iou_thres_;

        Ort::Env env_;
        Ort::SessionOptions sess_opts_;
        Ort::Session sess_;
        std::string input_name_;
        std::string output_name_;

        // 预处理：BGR -> RGB，resize，归一化，CHW -> NCHW
        cv::Mat preprocess(const cv::Mat& img_bgr, float& scale_x, float& scale_y);

        // NMS
        static float IoU(const cv::Rect& a, const cv::Rect& b);
        std::vector<Detection> nms(const std::vector<Detection>& dets);

    };

}