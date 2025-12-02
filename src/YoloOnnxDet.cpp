#include "YoloOnnxDet.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>

using namespace std;

namespace BodyHand {

    YoloOnnx::YoloOnnx(const std::string& onnx_path,
                    const cv::Size& input_size,
                    float conf_thres,
                    float iou_thres)
        : env_(ORT_LOGGING_LEVEL_WARNING, "yolov8"),
        input_size_(input_size),
        conf_thres_(conf_thres),
        iou_thres_(iou_thres),
        sess_(nullptr)
    {
        sess_opts_.SetIntraOpNumThreads(1);
        // 如需 CUDA EP，可在此添加配置
        sess_ = Ort::Session(env_, onnx_path.c_str(), sess_opts_);

        Ort::AllocatorWithDefaultOptions allocator;
        {
            Ort::AllocatedStringPtr in_name = sess_.GetInputNameAllocated(0, allocator);
            input_name_ = in_name.get();
        }
        {
            Ort::AllocatedStringPtr out_name = sess_.GetOutputNameAllocated(0, allocator);
            output_name_ = out_name.get();
        }
    }

    cv::Mat YoloOnnx::preprocess(const cv::Mat& img_bgr, float& scale_x, float& scale_y)
    {
        cv::Mat img_rgb;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

        int orig_h = img_rgb.rows;
        int orig_w = img_rgb.cols;

        cv::Mat resized;
        cv::resize(img_rgb, resized, input_size_);

        scale_x = static_cast<float>(orig_w) / input_size_.width;
        scale_y = static_cast<float>(orig_h) / input_size_.height;

        resized.convertTo(resized, CV_32F, 1.0 / 255.0);

        // HWC -> CHW
        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);

        cv::Mat chw(3, input_size_.width * input_size_.height, CV_32F);
        for (int i = 0; i < 3; ++i) {
            memcpy(chw.ptr<float>(i),
                channels[i].data,
                input_size_.width * input_size_.height * sizeof(float));
        }

        // [1,3,H,W]
        return chw.reshape(1, {1, 3, input_size_.height, input_size_.width});
    }

    float YoloOnnx::IoU(const cv::Rect& a, const cv::Rect& b)
    {
        int inter_x1 = max(a.x, b.x);
        int inter_y1 = max(a.y, b.y);
        int inter_x2 = min(a.x + a.width,  b.x + b.width);
        int inter_y2 = min(a.y + a.height, b.y + b.height);

        int inter_w = max(0, inter_x2 - inter_x1);
        int inter_h = max(0, inter_y2 - inter_y1);
        int inter_area = inter_w * inter_h;

        int union_area = a.width * a.height + b.width * b.height - inter_area;
        if (union_area <= 0) return 0.0f;
        return static_cast<float>(inter_area) / static_cast<float>(union_area);
    }

    std::vector<Detection> YoloOnnx::nms(const std::vector<Detection>& dets)
    {
        std::vector<Detection> result;
        std::vector<int> idxs(dets.size());
        std::iota(idxs.begin(), idxs.end(), 0);

        std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
            return dets[i].score > dets[j].score;
        });

        std::vector<bool> removed(dets.size(), false);
        for (size_t i = 0; i < idxs.size(); ++i) {
            int idx = idxs[i];
            if (removed[idx]) continue;
            const Detection& det_i = dets[idx];
            result.push_back(det_i);

            for (size_t j = i + 1; j < idxs.size(); ++j) {
                int idx_j = idxs[j];
                if (removed[idx_j]) continue;
                if (det_i.cls != dets[idx_j].cls) continue; // 类别不同可不互相抑制
                float iou = IoU(det_i.bbox, dets[idx_j].bbox);
                if (iou > iou_thres_) {
                    removed[idx_j] = true;
                }
            }
        }

        return result;
    }

    std::vector<Detection> YoloOnnx::infer(const cv::Mat& img_bgr,
                                        const std::vector<int>& allowed_cls)
    {
        float scale_x = 1.f, scale_y = 1.f;
        cv::Mat input_tensor = preprocess(img_bgr, scale_x, scale_y);

        std::array<int64_t, 4> input_shape = {1, 3, input_size_.height, input_size_.width};
        size_t tensor_size = 1ULL * 3 * input_size_.height * input_size_.width;

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtDeviceAllocator, OrtMemTypeCPU);

        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            mem_info,
            input_tensor.ptr<float>(),
            tensor_size,
            input_shape.data(),
            input_shape.size());

        const char* input_names[]  = { input_name_.c_str() };
        const char* output_names[] = { output_name_.c_str() };

        auto output_tensors = sess_.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_ort, 1,
            output_names, 1);

        float* out_data = output_tensors[0].GetTensorMutableData<float>();
        auto out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto out_shape = out_info.GetShape();
        // 预期: [1, 84, N] (4 + num_classes)
        if (out_shape.size() != 3 && out_shape.size() != 4) {
            throw std::runtime_error("Unsupported YOLO output shape");
        }

        int batch = static_cast<int>(out_shape[0]);
        if (batch != 1) {
            throw std::runtime_error("Only batch=1 is supported");
        }

        int C, N;
        if (out_shape.size() == 3) {
            C = static_cast<int>(out_shape[1]);
            N = static_cast<int>(out_shape[2]);
        } else { // [1, N, C]
            // 如你的模型是 [1, N, 84]，可在此调整解析方式
            C = static_cast<int>(out_shape[2]);
            N = static_cast<int>(out_shape[1]);
        }

        std::vector<Detection> dets;

        // 假设格式为 [1, C, N]，前4个通道为 box (cx,cy,w,h)，其余为各类别分数
        // 如果是 [1, N, C]，需要相应调整索引
        bool channels_first = (out_shape.size() == 3); // [1,C,N]

        int num_classes = C - 4;

        for (int i = 0; i < N; ++i) {
            float cx, cy, w, h;
            if (channels_first) {
                cx = out_data[0 * N + i];
                cy = out_data[1 * N + i];
                w  = out_data[2 * N + i];
                h  = out_data[3 * N + i];
            } else {
                // [1, N, C] 时的解析方式（按需改）
                int base = i * C;
                cx = out_data[base + 0];
                cy = out_data[base + 1];
                w  = out_data[base + 2];
                h  = out_data[base + 3];
            }

            // 取最大类别分数
            int best_cls = -1;
            float best_score = 0.0f;
            for (int cls = 0; cls < num_classes; ++cls) {
                float score;
                if (channels_first) {
                    score = out_data[(4 + cls) * N + i];
                } else {
                    int base = i * C;
                    score = out_data[base + 4 + cls];
                }
                if (score > best_score) {
                    best_score = score;
                    best_cls = cls;
                }
            }

            if (best_score < conf_thres_) continue;

            if (!allowed_cls.empty()) {
                if (std::find(allowed_cls.begin(), allowed_cls.end(), best_cls) == allowed_cls.end()) {
                    continue;
                }
            }

            // cx,cy,w,h -> x1,y1,x2,y2（在模型输入尺度上）
            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            // 缩放回原图
            x1 *= scale_x;
            x2 *= scale_x;
            y1 *= scale_y;
            y2 *= scale_y;

            int ix1 = std::max(0, static_cast<int>(std::round(x1)));
            int iy1 = std::max(0, static_cast<int>(std::round(y1)));
            int ix2 = std::min(img_bgr.cols - 1, static_cast<int>(std::round(x2)));
            int iy2 = std::min(img_bgr.rows - 1, static_cast<int>(std::round(y2)));

            int w_box = std::max(0, ix2 - ix1);
            int h_box = std::max(0, iy2 - iy1);
            if (w_box <= 0 || h_box <= 0) continue;

            Detection d;
            d.bbox = cv::Rect(ix1, iy1, w_box, h_box);
            d.score = best_score;
            d.id = -1;
            d.cls = best_cls;
            dets.push_back(d);
        }

        // NMS
        return nms(dets);
    }

}