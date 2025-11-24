#include <thread>
#include <chrono>
#include <filesystem>
#include <fmt/core.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "argparse.h"
#include "CMultiCap.h"
#include "stb_image_write.h"

int N_CAP = 2;

int main(int argc, char** argv) {
    argparse::ArgumentParser program("camera_capture");
    program.add_argument("--exposure")
        .help("Exposure time")
        .default_value(10000.f)
        .scan<'f', float>();
    program.add_argument("--gain")
        .help("Gain")
        .default_value(-1.f)
        .scan<'f', float>();
    program.add_argument("--output_dir")
        .help("Output directory.")
        .default_value(std::string("."));
    program.add_argument("--nosave")
        .help("Do not save the images.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--fps")
        .help("Framerate for capturing.")
        .default_value(25u)
        .scan<'u', unsigned int>();
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    // Retrieve values
    auto output_dir_str = program.get<std::string>("--output_dir");
    auto nosave = program.get<bool>("--nosave");
    auto exposure = program.get<float>("--exposure");
    auto gain = program.get<float>("--gain");
    auto fps = program.get<unsigned int>("--fps");

    auto capture_duration = 0; // in us
    if (fps != 0) {
        capture_duration = 1000000 / fps;
    }

    auto output_path = std::filesystem::path(output_dir_str);
    if (!std::filesystem::is_directory(output_path)) {
        std::cerr << "Output directory must be a directory." << std::endl;
        exit(-1);
    }
    if (output_path.is_relative()) {
        output_dir_str = std::filesystem::absolute(output_dir_str).string();
    }
    if (!nosave) {
        std::filesystem::create_directories(output_dir_str + "/V0");
        std::filesystem::create_directories(output_dir_str + "/V1");
    }

	get_app();
	init_device();

    if (get_device_count() < N_CAP) {
        std::cerr << "No enough device found." << std::endl;
        exit(-1);
    }

    for (int i = 0; i < N_CAP; ++i) {
        set_exposure(i, exposure);
        set_gain(i, gain);
    }
    std::cout << std::endl;
    if (!nosave) {
        std::cout << "Captured images will be saved to: " << output_dir_str << std::endl;
    }
    std::cout << "Setting exposure time to: " << exposure << "us" << std::endl;
    std::cout << "Setting gain to: " << gain << "\n" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));

    start_grabbing();

	for (int fx = 0;; ++fx) {
        auto start_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

		auto cap_info = capture();
		assert(cap_info.n_cap == N_CAP);

        for (int i = 0; i < N_CAP; ++i) {
            if (!cap_info.flag[i]) {
                continue;
            }
            cv::Mat img_rgb(cap_info.height[i], cap_info.width[i], CV_8UC3, cap_info.ppbuffer[i]), img_bgr;
            cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);
            cv::Mat img_demo;
            cv::resize(img_bgr, img_demo, cv::Size(0, 0), 0.5, 0.5);
            cv::imshow(fmt::format("{}", i), img_demo);

            if (!nosave) {
                std::string path = fmt::format(R"({}/V{}/{:06}.bmp)", output_dir_str, i, fx);
                stbi_write_bmp(path.c_str(), cap_info.width[0], cap_info.height[0], 3, cap_info.ppbuffer[i]);
            }
        }
        if (cv::waitKey(1) == 'q') {
            break;
        }

        auto end_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        auto cost_time = end_time - start_time;
        auto rest_time = capture_duration - cost_time;
        if (rest_time > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(rest_time));
        }
	}

    stop_grabbing();

	close_device();
	release_app();
	return 0;
}