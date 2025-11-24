// MultiCap.cpp
#include "MultiCap.h"
#include <cstring>
#include <algorithm>

MultiCap* multicap_obj = nullptr;

MultiCap* get_singleton() {
    if (!multicap_obj) multicap_obj = new MultiCap();
    return multicap_obj;
}

void release_singleton() {
    delete multicap_obj;
    multicap_obj = nullptr;
}

int MultiCap::init_device() {
    if (CMvCamera::EnumDevices(MV_USB_DEVICE, &device_info_list) != MV_OK
     || device_info_list.nDeviceNum == 0) return 0;

    for (unsigned i = 0; i < device_info_list.nDeviceNum; ++i) {
        auto* cam = new CMvCamera{};
        if (cam->Open(device_info_list.pDeviceInfo[i]) != MV_OK) { delete cam; continue; }
        cam->SetEnumValue("TriggerMode", 1);
        cam->SetEnumValue("TriggerSource", 7);
        cam->SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed);
        cam->SetBoolValue("GammaEnable", true);
        cam->SetEnumValue("GammaSelector", 2);
        device_obj_list.push_back(cam);
        frame_buffer_list.emplace_back();
        frame_flag.emplace_back(false);
    }
    return device_obj_list.size();
}

int MultiCap::start_grabbing() {
    b_grab = true;
    sem_init(&sem_agg, 0, 0);
    sem_init(&sem_continue, 0, 0);

    int grab_num = 0;
    for (unsigned i = 0; i < device_obj_list.size(); ++i) {
        if (device_obj_list[i]->StartGrabbing() != MV_OK) continue;
        ++grab_num;
        auto* ctrl = new SyncCtrl{this, i};
        pthread_t tid;
        if (pthread_create(&tid, nullptr, ::work_thread, ctrl) == 0) {
            thread_list.push_back(tid);
        } else {
            --grab_num;
            device_obj_list[i]->StopGrabbing();
            delete ctrl;
        }
    }
    return grab_num;
}

void* work_thread(void* p_user) {
    auto* st = reinterpret_cast<SyncCtrl*>(p_user);
    st->obj_ptr->_work_thread(st->cam_ix);
    delete st;
    return nullptr;
}

void MultiCap::_work_thread(unsigned int cam_ix) {
    auto* cam = device_obj_list[cam_ix];
    auto& buf = frame_buffer_list[cam_ix];

    MVCC_INTVALUE_EX size_info{0};
    if (cam->GetIntValue("PayloadSize", &size_info) != MV_OK) { sem_post(&sem_agg); return; }
    unsigned buf_size = size_info.nCurValue;
    buf.p_buffer = new unsigned char[buf_size];

    MV_FRAME_OUT_INFO_EX info{0};
    MV_CC_DEVICE_INFO dev_info{0};
    cam->GetDeviceInfo(&dev_info);

    while (b_grab) {
        sem_wait(&sem_continue);
        if (!b_grab) break;
        if (cam->GetOneFrameTimeout(buf.p_buffer, buf_size, &info, 1000) == MV_OK) {
            frame_flag[cam_ix] = true;
            buf.width   = info.nWidth;
            buf.height  = info.nHeight;
            buf.frame_ix= info.nFrameNum;
            std::strncpy(reinterpret_cast<char*>(buf.p_serial_number),
                         reinterpret_cast<const char*>(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber),
                         INFO_MAX_BUFFER_SIZE);
        }
        sem_post(&sem_agg);
    }
    delete[] buf.p_buffer;
}

void MultiCap::capture() {
    std::fill(frame_flag.begin(), frame_flag.end(), false);
    for (size_t i = 0; i < thread_list.size(); ++i) sem_post(&sem_continue);
    for (auto* cam : device_obj_list) cam->CommandExecute("TriggerSoftware");
    for (size_t i = 0; i < device_obj_list.size(); ++i) sem_wait(&sem_agg);
}

int MultiCap::stop_grabbing() {
    b_grab = false;
    for (size_t i = 0; i < thread_list.size(); ++i) sem_post(&sem_continue);
    int stopped = 0;
    for (auto* cam : device_obj_list) if (cam->StopGrabbing() == MV_OK) ++stopped;
    for (auto& tid : thread_list) pthread_join(tid, nullptr);
    sem_destroy(&sem_agg);
    sem_destroy(&sem_continue);
    return stopped;
}

int MultiCap::close_device() {
    int closed = 0;
    for (auto* cam : device_obj_list) if (cam->Close() == MV_OK) ++closed;
    device_obj_list.clear();
    return closed;
}

int MultiCap::set_exposure(unsigned int i, float exp) {
    if (i >= device_obj_list.size()) return -1;
    device_obj_list[i]->SetEnumValue("ExposureMode", 0);
    return device_obj_list[i]->SetFloatValue("ExposureTime", exp) == MV_OK ? 0 : -1;
}

int MultiCap::set_gain(unsigned int i, float gain) {
    if (i >= device_obj_list.size()) return -1;
    if (gain < 0) device_obj_list[i]->SetEnumValue("GainAuto", 1);
    else {
        device_obj_list[i]->SetEnumValue("GainAuto", 0);
        device_obj_list[i]->SetFloatValue("Gain", gain);
    }
    return 0;
}