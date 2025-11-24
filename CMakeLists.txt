cmake_minimum_required(VERSION 3.12)

project(BodyHand)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

# MvCamera用于相机捕捉
add_library(CMultiCap STATIC
    src/MvCamera.cpp
    src/MultiCap.cpp
    src/CMultiCap.cpp
)
target_include_directories(CMultiCap PUBLIC
    "include/"
    "/opt/MVS/include"
)
target_link_directories(CMultiCap PUBLIC
    "/opt/MVS/lib/64"
)
target_link_libraries(CMultiCap PUBLIC
    MvCameraControl
    pthread
)

# 用于图像写入
add_library(stb_image_write STATIC
    src/stb_image_write.cpp
)
target_include_directories(stb_image_write PUBLIC
    "include/"
)

# 文本格式化
add_subdirectory(third/fmt)

# SynchronizeCapture
add_executable(SynchronizeCapture
    src/main_SynchronizeCapture.cpp
)
target_include_directories(SynchronizeCapture PRIVATE
    "include/"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(SynchronizeCapture PRIVATE
    CMultiCap
    stb_image_write
    fmt::fmt
    ${OpenCV_LIBS}
)

# Calibratoin
add_executable(Calibration
    src/denoise.cpp
    src/detectChessboardCorner.cpp
    src/multiCameraBA.cpp
    src/multiCameraCalibration.cpp
    src/visualization.cpp
    src/main_Calibration.cpp
)
target_compile_definitions(Calibration PRIVATE GLOG_USE_GLOG_EXPORT)
target_include_directories(Calibration PRIVATE
    "include/"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(Calibration PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
)

# SingleCalibration
add_executable(SingleCalibration
    src/main_SingleCalibration.cpp
)
target_include_directories(SingleCalibration PRIVATE
    "include/"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(SingleCalibration PRIVATE
    ${OpenCV_LIBS}
)

# BodyHand
set(ORT_DIR "lib/onnxruntime-linux-x64-gpu-1.15.1")
add_library(BodyHand STATIC
    src/HaMeRONNX.cpp
    src/Pose3D.cpp
    src/YoloONNX.cpp
)
target_include_directories(BodyHand PRIVATE
    "include/"
    ${ORT_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(BodyHand PRIVATE
    ${ORT_DIR}/lib
)
target_link_libraries(BodyHand PRIVATE
    onnxruntime
    ${OpenCV_LIBS}
)

# main_MultiviewPoseEstimation
add_executable(MultiviewPoseEstimation
    src/main_MultiViewPoseEstimation.cpp
    src/tcp_server.cpp
)
target_include_directories(MultiviewPoseEstimation PRIVATE
    "include/"
    ${ORT_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(MultiviewPoseEstimation PRIVATE
    ${ORT_DIR}/lib
)
target_link_libraries(MultiviewPoseEstimation PRIVATE
    BodyHand
    CMultiCap
    onnxruntime
    ${OpenCV_LIBS}
)
