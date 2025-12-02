#include "VitInference.h"
#include <stdexcept>
#include <numeric>

namespace BodyHand {

    VitInference::VitInference(const std::string& pose_model,
                            const cv::Size& pose_input_size,
                            const std::string& device)
        : target_size_(pose_input_size),
        num_joints_(0),
        env_(ORT_LOGGING_LEVEL_WARNING, "vitpose"),
        sess_opts_(),
        sess_(nullptr),
        mean_{0.485f, 0.456f, 0.406f},
        std_{0.229f, 0.224f, 0.225f}
    {
        sess_opts_.SetIntraOpNumThreads(1);
        // 如需 CUDA EP，可在此添加（与你的 ORT GPU 版本匹配即可）
        // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sess_opts_, 0));

        sess_ = Ort::Session(env_, pose_model.c_str(), sess_opts_);

        Ort::AllocatorWithDefaultOptions allocator;
        {
            Ort::AllocatedStringPtr in_name = sess_.GetInputNameAllocated(0, allocator);
            input_name_ = in_name.get();
        }
        {
            Ort::AllocatedStringPtr out_name = sess_.GetOutputNameAllocated(0, allocator);
            output_name_ = out_name.get();
        }

        // 不再在这里调用 GetOutputTypeAndShapeInfo().GetShape()
        // num_joints_ 留到第一次 inference 时再根据实际输出确定

        // 简单 COCO skeleton (1-based -> 0-based)
        skeleton_ = {
            {0,1},{1,3},{0,2},{2,4},
            {3,5},{4,6},{5,6},{6,12},{12,11},{11,5},
            {6,8},{8,10},{5,7},{7,9},
            {12,14},{14,16},{11,13},{13,15}
        };
    }

    cv::Mat VitInference::padImage(const cv::Mat& img, float aspect, int& top, int& left)
    {
        int h = img.rows;
        int w = img.cols;

        float cur_aspect = static_cast<float>(w) / static_cast<float>(h);
        top = left = 0;
        int new_w = w, new_h = h;

        if (cur_aspect > aspect) {
            new_h = static_cast<int>(w / aspect + 0.5f);
            top = (new_h - h) / 2;
        } else {
            new_w = static_cast<int>(h * aspect + 0.5f);
            left = (new_w - w) / 2;
        }

        cv::Mat padded;
        cv::copyMakeBorder(img, padded,
                        top, new_h - h - top,
                        left, new_w - w - left,
                        cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        return padded;
    }

    cv::Mat VitInference::preImg(const cv::Mat& img, int& org_h, int& org_w)
    {
        org_h = img.rows;
        org_w = img.cols;

        cv::Mat resized;
        cv::resize(img, resized, target_size_, 0, 0, cv::INTER_LINEAR); // RGB

        resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);

        cv::Mat chw(3, target_size_.width * target_size_.height, CV_32F);
        for (int i = 0; i < 3; ++i) {
            channels[i] = (channels[i] - mean_[i]) / std_[i];
            memcpy(chw.ptr<float>(i),
                channels[i].data,
                target_size_.width * target_size_.height * sizeof(float));
        }

        return chw.reshape(1, {1, 3, target_size_.height, target_size_.width});
    }

    cv::Mat VitInference::postprocess(const std::vector<float>& heatmaps,
                                    int c, int h, int w,
                                    int org_w, int org_h)
    {
        cv::Mat keypoints(c, 3, CV_32F);

        for (int j = 0; j < c; ++j) {
            const float* hm = heatmaps.data() + j * h * w;
            int   max_idx = 0;
            float max_val = hm[0];
            for (int i = 1; i < h * w; ++i) {
                if (hm[i] > max_val) {
                    max_val = hm[i];
                    max_idx = i;
                }
            }
            int yy = max_idx / w;
            int xx = max_idx % w;

            float x = static_cast<float>(xx) / (w - 1) * (org_w - 1);
            float y = static_cast<float>(yy) / (h - 1) * (org_h - 1);

            keypoints.at<float>(j, 0) = x;
            keypoints.at<float>(j, 1) = y;
            keypoints.at<float>(j, 2) = max_val;
        }

        return keypoints;
    }

    std::map<int, cv::Mat> VitInference::inference(const cv::Mat& img_rgb,
                                                const std::vector<Detection>& dets)
    {
        std::map<int, cv::Mat> frame_keypoints;

        for (size_t idx = 0; idx < dets.size(); ++idx) {
            const auto& det = dets[idx];

            cv::Rect box = det.bbox & cv::Rect(0, 0, img_rgb.cols, img_rgb.rows);
            if (box.width <= 0 || box.height <= 0) continue;

            cv::Mat crop = img_rgb(box).clone();

            int top_pad = 0, left_pad = 0;
            cv::Mat padded = padImage(crop, 3.0f / 4.0f, top_pad, left_pad);

            int org_h, org_w;
            cv::Mat input_tensor = preImg(padded, org_h, org_w);

            std::array<int64_t,4> input_shape = {1, 3, target_size_.height, target_size_.width};
            size_t tensor_size = 1ULL * 3 * target_size_.height * target_size_.width;

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
            auto out_info   = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto out_shape  = out_info.GetShape(); // [1,C,H,W]

            if (out_shape.size() != 4) {
                throw std::runtime_error("ViTPose runtime output shape != 4D");
            }

            int C = static_cast<int>(out_shape[1]);
            int H = static_cast<int>(out_shape[2]);
            int W = static_cast<int>(out_shape[3]);
            size_t out_size = 1ULL * C * H * W;

            if (num_joints_ == 0) {
                num_joints_ = C; // 第一次推理时记录关节数
            }

            std::vector<float> hm(out_data, out_data + out_size);

            cv::Mat kpts = postprocess(hm, C, H, W, padded.cols, padded.rows);

            for (int i = 0; i < kpts.rows; ++i) {
                float& x = kpts.at<float>(i, 0);
                float& y = kpts.at<float>(i, 1);
                x = x + box.x - left_pad;
                y = y + box.y - top_pad;
            }

            int id = (det.id < 0) ? static_cast<int>(idx) : det.id;
            frame_keypoints[id] = kpts;
        }

        return frame_keypoints;
    }

    cv::Mat VitInference::draw(const cv::Mat& img_rgb,
                            const std::map<int, cv::Mat>& kpts_map,
                            const std::vector<Detection>& dets,
                            float kp_conf_thr,
                            bool show_yolo)
    {
        cv::Mat vis;
        img_rgb.copyTo(vis);

        if (show_yolo) {
            for (const auto& det : dets) {
                cv::rectangle(vis, det.bbox, cv::Scalar(0, 255, 0), 2);
            }
        }

        for (const auto& it : kpts_map) {
            const cv::Mat& kpts = it.second;

        for (int i = 0; i < kpts.rows; ++i) {
            float x = kpts.at<float>(i, 0);
            float y = kpts.at<float>(i, 1);
            float s = kpts.at<float>(i, 2);
            if (s < kp_conf_thr) continue;

            // 绘制圆点
            cv::circle(vis, cv::Point2f(x, y), 3, cv::Scalar(0, 0, 255), -1);

            // 绘制序号（白字黑边，更清晰）
            char buf[8];
            snprintf(buf, sizeof(buf), "%d", i);
            cv::putText(vis, buf,
                        cv::Point2f(x + 4, y - 4),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(vis, buf,
                        cv::Point2f(x + 4, y - 4),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1,
                        cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        }

            for (auto p : skeleton_) {
                int i1 = p.first;
                int i2 = p.second;
                if (i1 < 0 || i1 >= kpts.rows || i2 < 0 || i2 >= kpts.rows) continue;

                float x1 = kpts.at<float>(i1, 0);
                float y1 = kpts.at<float>(i1, 1);
                float s1 = kpts.at<float>(i1, 2);
                float x2 = kpts.at<float>(i2, 0);
                float y2 = kpts.at<float>(i2, 1);
                float s2 = kpts.at<float>(i2, 2);

                if (s1 < kp_conf_thr || s2 < kp_conf_thr) continue;

                cv::line(vis, cv::Point2f(x1, y1), cv::Point2f(x2, y2),
                        cv::Scalar(255, 0, 0), 2);
            }
        }

        return vis;
    }

}