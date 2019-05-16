# Stereolabs ZED - YOLO 3D

This package lets you use [YOLO (v2 or v3)](http://pjreddie.com/darknet/yolo/), the deep learning object detector using the ZED stereo camera in Python 3 or C++.

# [Update : the ZED is now natively supported in YOLO !](https://github.com/AlexeyAB/darknet)

## 1. Setup

The setup detailed setup instructions are available in the [Darknet repository](https://github.com/AlexeyAB/darknet).

This is a brief explanation on how to enable the ZED camera support.

### Prerequisites

- Windows 7 64bits or later, Ubuntu 16.04 or 18.04
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- **Darknet** : https://github.com/AlexeyAB/darknet

## Installing Darknet

[Download](https://github.com/AlexeyAB/darknet) and compile darknet, following the instructions:

- [How to compile on Linux](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux)
- [How to compile on Windows](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows-using-vcpkg)

### ZED Support Using CMake

If the ZED SDK is installed, CMake will automatically detect it and compile with the ZED support. During the CMake configuration, a message will confirm that the ZED SDK was found.

    ...
    -- A library with BLAS API found.
    -- ZED SDK enabled
    -- Found OpenMP_C: -fopenmp (found version "4.5")
    ...

### ZED support Using Makefile

To enable the ZED support in YOLO using the Makefile, simply enable [`GPU` and `ZED_CAMERA`](https://github.com/AlexeyAB/darknet/blob/cce34712f6928495f1fbc5d69332162fc23491b9/Makefile#L8), it's also recommended to enable `CUDNN` for improved performances.

## 2. Launching the sample

    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights zed_camera

SVO files are also supported :

    ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights /path/to/svo/file.svo

## How to use YOLO 3D in Python

The native support is currently only in C++.

For the Python version please refer to instructions in [zed_python_sample](./zed_python_sample)

## Using Docker

A DockerFile is provided in the [docker folder](./docker)

## Legacy repository

The original YOLO 3D C++ sources are available in the [legacy branch](https://github.com/stereolabs/zed-yolo/tree/legacy)
