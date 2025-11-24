// MultiCap.h
#ifndef MULTICAP_H
#define MULTICAP_H

#include "MvCamera.h"
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include <atomic>
#include <cstring>

struct MultiCap;

struct SyncCtrl { MultiCap* obj_ptr; unsigned int cam_ix; };

struct BufferInfo {
    unsigned int width;
    unsigned int height;
    unsigned int frame_ix;
    unsigned char* p_buffer;
    unsigned char p_serial_number[INFO_MAX_BUFFER_SIZE];
};

struct MultiCap {
    MV_CC_DEVICE_INFO_LIST device_info_list{0};
    std::vector<CMvCamera*> device_obj_list;
    std::vector<BufferInfo>   frame_buffer_list;
    std::vector<bool>         frame_flag;
    std::vector<pthread_t>    thread_list;

    sem_t sem_agg, sem_continue;
    std::atomic_bool b_grab{false};

    MultiCap() = default;
    ~MultiCap() = default;

    int  init_device();
    int  start_grabbing();
    void _work_thread(unsigned int cam_ix);
    void capture();
    int  stop_grabbing();
    int  close_device();
    int  get_device_count() const { return device_obj_list.size(); }
    int  set_exposure(unsigned int cam_ix, float exposure);
    int  set_gain(unsigned int cam_ix, float gain);
};

extern MultiCap* multicap_obj;
MultiCap* get_singleton();
void      release_singleton();
void*     work_thread(void* p_user);

#endif // MULTICAP_H
