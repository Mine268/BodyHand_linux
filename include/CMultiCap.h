#ifndef HEAD_CMULTICAP
#define HEAD_CMULTICAP

#if defined(_WIN32) || defined(_WIN64)
  #define EXPORT_API __declspec(dllexport)
#else
  #define EXPORT_API __attribute__((visibility("default")))
#endif

constexpr unsigned int INFO_MAX_BUFFER_SIZE = 64;
constexpr unsigned int MAX_CAPTURE = 16;


extern "C" {

    struct CaptureInfo {
        unsigned int n_cap;  // 捕捉的数量
        bool flag[MAX_CAPTURE];  // 每一个捕捉是否成功
        unsigned int width[MAX_CAPTURE];  // 每一个捕捉的宽度
        unsigned int height[MAX_CAPTURE];  // 每一个图像的高度
        unsigned char* ppbuffer[MAX_CAPTURE];  // 每一个捕捉的缓冲区的指针
        unsigned char serial_numbers[MAX_CAPTURE][INFO_MAX_BUFFER_SIZE]; // 每一个捕捉的图像对应相机的序列号
    };

    // 获得单例
    EXPORT_API void get_app();

    // 关系系统
    EXPORT_API void release_app();

    // 获取设备
    EXPORT_API auto init_device() -> int;

    // 关闭设备
    EXPORT_API auto close_device() -> int;

    // 开始抓取
    EXPORT_API auto start_grabbing() -> int;

    // 停止抓取
    EXPORT_API auto stop_grabbing() -> int;

    // 抓取一帧
    EXPORT_API auto capture() -> CaptureInfo;

    // 获取设备数量
    EXPORT_API auto get_device_count() -> int;

    // 设置设备曝光
    EXPORT_API auto set_exposure(unsigned cam_ix, float exposure) -> int;

    // 设置设备增益
    EXPORT_API auto set_gain(unsigned cam_ix, float gain) -> int;

}


#endif // !HEAD_CMULTICAP