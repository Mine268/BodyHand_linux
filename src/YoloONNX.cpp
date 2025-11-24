// Ref: https://blog.csdn.net/weixin_45824067/article/details/130618583

#include "YoloONNX.h"

namespace BodyHand {

    using namespace std;
    using namespace cv;
    using namespace cv::dnn;
    using namespace Ort;

    void LetterBox(
        const cv::Mat& image, cv::Mat& outImage,
        cv::Vec4d& params,
        const cv::Size& newShape,
        bool autoShape,
        bool scaleFill,
        bool scaleUp,
        int stride,
        const cv::Scalar& color
    ) {
        if (false) {
            int maxLen = MAX(image.rows, image.cols);
            outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
            image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
            params[0] = 1;
            params[1] = 1;
            params[3] = 0;
            params[2] = 0;
        }

        // 取较小的缩放比例
        cv::Size shape = image.size();
        float r = std::min((float)newShape.height / (float)shape.height,
            (float)newShape.width / (float)shape.width);
        if (!scaleUp)
            r = std::min(r, 1.0f);
        //printf("原图尺寸：w:%d * h:%d, 要求尺寸：w:%d * h:%d, 即将采用的缩放比：%f\n",
        //	shape.width, shape.height, newShape.width, newShape.height, r);

        // 依据前面的缩放比例后，原图的尺寸
        float ratio[2]{ r,r };
        int new_un_pad[2] = { (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r) };
        //printf("等比例缩放后的尺寸该为：w:%d * h:%d\n", new_un_pad[0], new_un_pad[1]);

        // 计算距离目标尺寸的padding像素数
        auto dw = (float)(newShape.width - new_un_pad[0]);
        auto dh = (float)(newShape.height - new_un_pad[1]);
        if (autoShape)
        {
            dw = (float)((int)dw % stride);
            dh = (float)((int)dh % stride);
        }
        else if (scaleFill)
        {
            dw = 0.0f;
            dh = 0.0f;
            new_un_pad[0] = newShape.width;
            new_un_pad[1] = newShape.height;
            ratio[0] = (float)newShape.width / (float)shape.width;
            ratio[1] = (float)newShape.height / (float)shape.height;
        }

        dw /= 2.0f;
        dh /= 2.0f;
        //printf("填充padding: dw=%f , dh=%f\n", dw, dh);

        // 等比例缩放
        if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
        {
            cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
        }
        else {
            outImage = image.clone();
        }

        // 图像四周padding填充，至此原图与目标尺寸一致
        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));
        params[0] = ratio[0]; // width的缩放比例
        params[1] = ratio[1]; // height的缩放比例
        params[2] = left; // 水平方向两边的padding像素数
        params[3] = top; //垂直方向两边的padding像素数
        cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    }

    bool Yolov8Onnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaId, bool warmUp) {
        if (_batchSize < 1) _batchSize = 1;
        try
        {
            std::vector<std::string> available_providers = GetAvailableProviders();
            auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");


            if (isCuda && (cuda_available == available_providers.end()))
            {
                std::cout << "Pose：无可用cuda，使用cpu进行推理。" << endl;
            }
            else if (isCuda && (cuda_available != available_providers.end()))
            {
                std::cout << "Pose：尝试使用cuda推理。" << std::endl;
                OrtCUDAProviderOptions cudaOption;
                cudaOption.device_id = cudaId;
                _OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
                //#if ORT_API_VERSION < ORT_OLD_VISON
                //	OrtCUDAProviderOptions cudaOption;
                //	cudaOption.device_id = cudaID;
                //    _OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
                //#else
                //	OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaId);
                //#endif
            }
            else
            {
                //std::cout << "Pose：使用CPU推理" << std::endl;
            }

            _OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
            std::wstring model_path(modelPath.begin(), modelPath.end());
            _OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
            _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

            Ort::AllocatorWithDefaultOptions allocator;
            //init input
            _inputNodesNum = _OrtSession->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
            _inputName = _OrtSession->GetInputName(0, allocator);
            _inputNodeNames.push_back(_inputName);
#else
            _inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
            _inputNodeNames.push_back(_inputName.get());
#endif
            //cout << _inputNodeNames[0] << endl;
            Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
            auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
            _inputNodeDataType = input_tensor_info.GetElementType();
            _inputTensorShape = input_tensor_info.GetShape();

            if (_inputTensorShape[0] == -1)
            {
                _isDynamicShape = true;
                _inputTensorShape[0] = _batchSize;

            }
            if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
                _isDynamicShape = true;
                _inputTensorShape[2] = _netHeight;
                _inputTensorShape[3] = _netWidth;
            }
            //init output
            _outputNodesNum = _OrtSession->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
            _output_name0 = _OrtSession->GetOutputName(0, allocator);
            _outputNodeNames.push_back(_output_name0);
#else
            _output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
            _outputNodeNames.push_back(_output_name0.get());
#endif
            Ort::TypeInfo type_info_output0(nullptr);
            type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0

            auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
            _outputNodeDataType = tensor_info_output0.GetElementType();
            _outputTensorShape = tensor_info_output0.GetShape();

            //_outputMaskNodeDataType = tensor_info_output1.GetElementType(); //the same as output0
            //_outputMaskTensorShape = tensor_info_output1.GetShape();
            //if (_outputTensorShape[0] == -1)
            //{
            //	_outputTensorShape[0] = _batchSize;
            //	_outputMaskTensorShape[0] = _batchSize;
            //}
            //if (_outputMaskTensorShape[2] == -1) {
            //	//size_t ouput_rows = 0;
            //	//for (int i = 0; i < _strideSize; ++i) {
            //	//	ouput_rows += 3 * (_netWidth / _netStride[i]) * _netHeight / _netStride[i];
            //	//}
            //	//_outputTensorShape[1] = ouput_rows;

            //	_outputMaskTensorShape[2] = _segHeight;
            //	_outputMaskTensorShape[3] = _segWidth;
            //}

            //warm up
            if (isCuda && warmUp) {
                //draw run
                //cout << "Start warming up" << endl;
                size_t input_tensor_length = VectorProduct(_inputTensorShape);
                float* temp = new float[input_tensor_length];
                std::vector<Ort::Value> input_tensors;
                std::vector<Ort::Value> output_tensors;
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    _OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
                    _inputTensorShape.size()));
                for (int i = 0; i < 3; ++i) {
                    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
                        _inputNodeNames.data(),
                        input_tensors.data(),
                        _inputNodeNames.size(),
                        _outputNodeNames.data(),
                        _outputNodeNames.size());
                }

                delete[]temp;
            }
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return false;
        }
        return true;

    }

    int Yolov8Onnx::Preprocessing(const std::vector<cv::Mat>& SrcImgs,
        std::vector<cv::Mat>& OutSrcImgs,
        std::vector<cv::Vec4d>& params) {
        OutSrcImgs.clear();
        Size input_size = Size(_netWidth, _netHeight);

        // 信封处理
        for (size_t i = 0; i < SrcImgs.size(); ++i) {
            Mat temp_img = SrcImgs[i];
            Vec4d temp_param = { 1,1,0,0 };
            if (temp_img.size() != input_size) {
                Mat borderImg;
                LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
                OutSrcImgs.push_back(borderImg);
                params.push_back(temp_param);
            }
            else {
                OutSrcImgs.push_back(temp_img);
                params.push_back(temp_param);
            }
        }

        int lack_num = _batchSize - SrcImgs.size();
        if (lack_num > 0) {
            Mat temp_img = Mat::zeros(input_size, CV_8UC3);
            Vec4d temp_param = { 1,1,0,0 };
            OutSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
        return 0;
    }

    // 这个方法的返回值表示检测结果中是否有人，但是存在 bug，应该直接判断数组长度决定
    bool Yolov8Onnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputPose>>& output)
    {
        vector<Vec4d> params;
        vector<Mat> input_images;
        cv::Size input_size(_netWidth, _netHeight);

        //preprocessing (信封处理)
        Preprocessing(srcImgs, input_images, params);
        // [0~255] --> [0~1]; BGR2RGB
        Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

        // 前向传播得到推理结果
        int64_t input_tensor_length = VectorProduct(_inputTensorShape);// ?
        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data,
            input_tensor_length, _inputTensorShape.data(),
            _inputTensorShape.size()));

        output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
            _inputNodeNames.data(),
            input_tensors.data(),
            _inputNodeNames.size(),
            _outputNodeNames.data(),
            _outputNodeNames.size()
        );

        //post-process

        float* all_data = output_tensors[0].GetTensorMutableData<float>(); // 第一张图片的输出

        _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); // 一张图片输出的维度信息 [1, 84, 8400]

        int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0]; // 一张图片输出所占内存长度 8400*84

        for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
            Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t(); // [1, 56 ,8400] -> [1, 8400, 56]

            all_data += one_output_length; //指针指向下一个图片的地址

            float* pdata = (float*)output0.data; // [classid,x,y,w,h,x,y,...21个点]
            int rows = output0.rows; // 预测框的数量 8400

            // 一张图片的预测框

            vector<float> confidences;
            vector<Rect> boxes;
            vector<int> labels;
            vector<vector<float>> kpss;
            for (int r = 0; r < rows; ++r) {

                // 得到人类别概率
                auto kps_ptr = pdata + 5;


                // 预测框坐标映射到原图上
                float score = pdata[4];
                if (score > _classThreshold) {

                    // rect [x,y,w,h]
                    float x = (pdata[0] - params[img_index][2]) / params[img_index][0]; //x
                    float y = (pdata[1] - params[img_index][3]) / params[img_index][1]; //y
                    float w = pdata[2] / params[img_index][0]; //w
                    float h = pdata[3] / params[img_index][1]; //h

                    int left = MAX(int(x - 0.5 * w + 0.5), 0);
                    int top = MAX(int(y - 0.5 * h + 0.5), 0);

                    std::vector<float> kps;
                    for (int k = 0; k < 17; k++) {
                        float kps_x = (*(kps_ptr + 3 * k) - params[img_index][2]) / params[img_index][0];
                        float kps_y = (*(kps_ptr + 3 * k + 1) - params[img_index][3]) / params[img_index][1];
                        float kps_s = *(kps_ptr + 3 * k + 2);

                        // cout << *(kps_ptr + 3*k) << endl;

                        kps.push_back(kps_x);
                        kps.push_back(kps_y);
                        kps.push_back(kps_s);
                    }

                    confidences.push_back(score);
                    labels.push_back(0);
                    kpss.push_back(kps);
                    boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
                }
                pdata += _anchorLength; //下一个预测框
            }

            // 对一张图的预测框执行NMS处理
            vector<int> nms_result;
            cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThrehold, nms_result); // 还需要classThreshold？

            // 对一张图片：依据NMS处理得到的索引，得到类别id、confidence、box，并置于结构体OutputDet的容器中
            vector<OutputPose> temp_output;
            for (size_t i = 0; i < nms_result.size(); ++i) {
                int idx = nms_result[i];
                OutputPose result;

                result.confidence = confidences[idx];
                result.box = boxes[idx];
                result.label = labels[idx];
                result.kps = kpss[idx];
                temp_output.push_back(result);
            }
            output.push_back(temp_output); // 多张图片的输出；添加一张图片的输出置于此容器中
        }
        if (output.size())
            return true;
        else
            return false;

    }


    bool Yolov8Onnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputPose>& output) {
        vector<Mat> input_data = { srcImg };
        vector<vector<OutputPose>> temp_output;

        if (OnnxBatchDetect(input_data, temp_output)) {
            output = temp_output[0];
            return true;
        }
        else return false;
    }

}