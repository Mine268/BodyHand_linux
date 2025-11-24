#include "CMultiCap.h"
#include "MultiCap.h"

extern "C" {

    EXPORT_API void get_app() {
        get_singleton();
    }

    EXPORT_API void release_app() {
        release_singleton();
    }

    EXPORT_API auto init_device() -> int {
        return multicap_obj->init_device();
    }

    EXPORT_API auto close_device() -> int {
        return multicap_obj->close_device();
    }

    EXPORT_API auto start_grabbing() -> int {
        return multicap_obj->start_grabbing();
    }

    EXPORT_API auto stop_grabbing() -> int {
        return multicap_obj->stop_grabbing();
    }

    EXPORT_API auto capture() -> CaptureInfo {
        CaptureInfo cap_info{};

        multicap_obj->capture();
        cap_info.n_cap = multicap_obj->device_obj_list.size();
        for (unsigned int cap_ix = 0; cap_ix < multicap_obj->frame_buffer_list.size(); ++cap_ix) {
            if (cap_ix >= MAX_CAPTURE) {
                break;
            }

            cap_info.flag[cap_ix] = multicap_obj->frame_flag[cap_ix];
            cap_info.width[cap_ix] = multicap_obj->frame_buffer_list[cap_ix].width;
            cap_info.height[cap_ix] = multicap_obj->frame_buffer_list[cap_ix].height;
            cap_info.ppbuffer[cap_ix] = multicap_obj->frame_buffer_list[cap_ix].p_buffer;
            {
                const char *src = reinterpret_cast<const char *>(
                    multicap_obj->frame_buffer_list[cap_ix].p_serial_number);
                // 拷 INFO_MAX_BUFFER_SIZE 个字节，最后一位一定要置 0
                std::strncpy(reinterpret_cast<char *>(cap_info.serial_numbers[cap_ix]),
                             src,
                             INFO_MAX_BUFFER_SIZE);
                cap_info.serial_numbers[cap_ix][INFO_MAX_BUFFER_SIZE - 1] = '\0';
            }
        }

        return cap_info;
    }

    EXPORT_API auto get_device_count() -> int {
        return multicap_obj->get_device_count();
    }

    EXPORT_API auto set_exposure(unsigned cam_ix, float exposure) -> int {
        return multicap_obj->set_exposure(cam_ix, exposure);
    }

    EXPORT_API auto set_gain(unsigned cam_ix, float gain) -> int {
        return multicap_obj->set_gain(cam_ix, gain);
    }

}
