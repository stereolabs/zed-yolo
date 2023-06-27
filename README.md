# Using YOLO and the ZED

For a detailed explanation please refer to the documentation https://www.stereolabs.com/docs/yolo/

This repository contains two samples to use YOLO with the ZED in C++ using the highly optimized library TensorRT, and a Python sample that uses Pytorch and the official package of ultralytics for YOLOv8

Other sample using OpenCV DNN or YOLOv5 using the TensorRT API in [C++](https://github.com/stereolabs/zed-sdk/tree/master/object%20detection/custom%20detector/cpp) or [Pytorch](https://github.com/stereolabs/zed-sdk/tree/master/object%20detection/custom%20detector/python) can be found in the main [ZED SDK repository](https://github.com/stereolabs/zed-sdk/tree/master/object%20detection/custom%20detector/cpp)


## YOLO v5, v6 or v8 using TensorRT and C++

In the folder [tensorrt_yolov5-v6-v8_onnx](./tensorrt_yolov5-v6-v8_onnx) you will find a sample that is able to run an ONNX model exported from YOLO architecture and using it with the ZED.

This sample is designed to run a state of the art object detection model using the highly optimized TensorRT framework. The image are taken from the ZED SDK, and the 2D box detections are then ingested into the ZED SDK to extract 3D informations (localization, 3D bounding boxes) and tracking.

This sample is using a TensorRT optimized ONNX model. It is compatible with YOLOv8, YOLOv5 and YOLOv6. It can be used with the default model trained on COCO dataset (80 classes) provided by the framework maintainers.

A custom detector can be trained with the same architecture.


## YOLOv8 Pytorch

In the folder [pytorch_yolov8](./pytorch_yolov8) you will find a sample that interface PyTorch ultralytics package with the ZED SDK in Python.

This sample shows how to detect custom objects using the official Pytorch implementation of YOLOv8 from a ZED camera and ingest them into the ZED SDK to extract 3D informations and tracking for each objects.