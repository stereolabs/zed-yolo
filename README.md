# Stereolabs ZED - YOLO 3D

This package lets you use [YOLO (v2, v3 or v4)](http://pjreddie.com/darknet/yolo/), the deep learning object detector using the ZED stereo camera in Python 3 or C++.

## The ZED SDK now support external detector for 3D object detection and tracking, more informations here:
- [ZED SDK custom object module documentation](https://www.stereolabs.com/docs/object-detection/custom-od/)
- [ZED SDK custom object examples](https://github.com/stereolabs/zed-examples/tree/master/object%20detection/custom%20detector)
- [ZED SDK with OpenCV DNN Yolov4 input](https://github.com/stereolabs/zed-examples/tree/master/object%20detection/custom%20detector/cpp/opencv_dnn_yolov4)

## 1. Setup

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [OpenCV](https://docs.opencv.org/4.x/da/df6/tutorial_py_table_of_contents_setup.html) built with CUDA, [cuDNN](https://developer.nvidia.com/cudnn) and DNN module

### Preparing OpenCV installation

**This version uses OpenCV DNN module for inference.**

### cuDNN

In order to get the best performance, [cuDNN](https://developer.nvidia.com/cudnn) should be installed before building OpenCV. Heads over to this [TensorFlow documentation article](https://www.tensorflow.org/install/gpu#install_cuda_with_apt) which explains how to setup both CUDA and cuDNN on Ubuntu and Windows.

### OpenCV DNN

Install OpenCV with DNN module enabled, it's advised to compile it from source to make sure the correct options are enabled.


## 2. Launching the YOLO 3D in C++

Download the yolo weights, [yolov4](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) for instance, and put them in the local folder.

```sh
cd zed_cpp_sample/

mkdir build
cd build
cmake ..
make

./build/yolo_zed 
```

## 3. Launching the YOLO 3D in Python

Download the yolo weights, [yolov4](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) for instance, and put them in the local folder.

```sh
cd zed_python_sample/
python3 zed_yolo.py
```

## Using Docker

A DockerFile is provided in the [docker folder](./docker)

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/
