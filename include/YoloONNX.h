#pragma once
// Ref: https://blog.csdn.net/weixin_45824067/article/details/130618583
// Ref: https://blog.csdn.net/weixin_43013458/article/details/144830905

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <numeric>

namespace BodyHand {

    struct OutputPose {
        cv::Rect_<float> box;
        int label = 0;
        float confidence = 0.0;
        std::vector<float> kps;
    };

    void LetterBox(
        const cv::Mat& image,
        cv::Mat& outImage,
        cv::Vec4d& params,
        const cv::Size& newShape = cv::Size(640, 640),
        bool autoShape = false,
        bool scaleFill = false,
        bool scaleUp = true,
        int stride = 32,
        const cv::Scalar& color = cv::Scalar(114, 114, 114)
    );

    class Yolov8Onnx {
    private:
        template<typename T>
        T VectorProduct(const std::vector<T>& v)
        {
            return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
        }

        int Preprocessing(
            const std::vector<cv::Mat>& SrcImgs,
            std::vector<cv::Mat>& OutSrcImgs,
            std::vector<cv::Vec4d>& params
        );

        const int _netWidth = 640;   //ONNX-net-input-width
        const int _netHeight = 640;  //ONNX-net-input-height

        int _batchSize = 2; //if multi-batch,set this
        bool _isDynamicShape = false;//onnx support dynamic shape

        int _anchorLength = 56;// pose一个框的信息56个数

        float _classThreshold = 0.25;
        float _nmsThrehold = 0.45;

        //ONNXRUNTIME
        Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov5-Seg");
        Ort::SessionOptions _OrtSessionOptions = Ort::SessionOptions();
        Ort::Session* _OrtSession = nullptr;
        Ort::MemoryInfo _OrtMemoryInfo;

        std::shared_ptr<char> _inputName, _output_name0;
        std::vector<char*> _inputNodeNames; //输入节点名
        std::vector<char*> _outputNodeNames; // 输出节点名

        size_t _inputNodesNum = 0;        // 输入节点数
        size_t _outputNodesNum = 0;      // 输出节点数

        ONNXTensorElementDataType _inputNodeDataType;  //数据类型
        ONNXTensorElementDataType _outputNodeDataType;

        std::vector<int64_t> _inputTensorShape;  // 输入张量shape
        std::vector<int64_t> _outputTensorShape;

    public:
        Yolov8Onnx() :_OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {};
        ~Yolov8Onnx() {};// delete _OrtMemoryInfo;

    public:
        /** \brief Read onnx-model
        * \param[in] modelPath:onnx-model path
        * \param[in] isCuda:if true,use Ort-GPU,else run it on cpu.
        * \param[in] cudaID:if isCuda==true,run Ort-GPU on cudaID.
        * \param[in] warmUp:if isCuda==true,warm up GPU-model.
        */
        bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaId = 0, bool warmUp = true);

        /** \brief  detect.
        * \param[in] srcImg:a 3-channels image.
        * \param[out] output:detection results of input image.
        */
        bool OnnxDetect(cv::Mat& srcImg, std::vector<OutputPose>& output);

        /** \brief  detect,batch size= _batchSize
        * \param[in] srcImg:A batch of images.
        * \param[out] output:detection results of input images.
        */
        bool OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputPose>>& output);

        //public:
        //    std::vector<std::string> _className = {
        //        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        //        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        //        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        //        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        //        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        //        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        //        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        //        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        //        "hair drier", "toothbrush"
        //    };

    };

}