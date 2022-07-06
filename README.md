# Stereolabs ZED - YOLO 3D

This package lets you use [YOLO (v2, v3 or v4)](http://pjreddie.com/darknet/yolo/), the deep learning object detector using the ZED stereo camera in Python 3 or C++.

# [Update : the ZED is now natively supported in YOLO !](https://github.com/AlexeyAB/darknet)

## 1. Setup

### Prerequisites

- Windows 7 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [OpenCV](https://docs.opencv.org/4.x/da/df6/tutorial_py_table_of_contents_setup.html) built with CUDA and [cuDNN](https://developer.nvidia.com/cudnn)

### Preparing OpenCV installation

### cuDNN

In order to get the best performance, [cuDNN](https://developer.nvidia.com/cudnn) should be install before building OpenCV. Heads over to this [TensorFlow documentation article](https://www.tensorflow.org/install/gpu#install_cuda_with_apt) which explains how to setup both CUDA and cuDNN on Ubuntu and Windows.

### OpenCV

OpenCV binaries can be downloaded and install from [opencv.org](https://opencv.org/releases/).

Alternatively, on Ubuntu :

    sudo apt install pkg-config libopencv-dev

### CMake

On Windows, download and install CMAKE using the binary [available here](https://cmake.org/download/).

On Ubuntu, cmake can be installed using the package manager, i.e : `sudo apt install cmake`

However the default version of cmake might be too old, it can easily be updated using the script (located in this repository):

```bash
sudo bash cmake_apt_update.sh
```

### ZED Support Using CMake (recommended)

If the ZED SDK is installed, CMake will automatically detect it and compile with the ZED support. During the CMake configuration, a message will confirm that the ZED SDK was found.

    ...
    -- A library with BLAS API found.
    -- ZED SDK enabled
    -- Found OpenMP_C: -fopenmp (found version "4.5")
    ...


## 2. Launching the YOLO 3D in C++

Download the yolo weights, [yolov4](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) for instance, and put them in the local folder.

    cd zed_cpp_sample/
    
    mkdir build
    cd build
    cmake ..
    make
    
    ./build/yolo_zed 

## 3. Launching the YOLO 3D in Python

Download the yolo weights, [yolov4](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) for instance, and put them in the local folder.

    cd zed_python_sample/
    
    python3 zed_yolo.py

## Using Docker

A DockerFile is provided in the [docker folder](./docker)

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/
