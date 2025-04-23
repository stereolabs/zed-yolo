# Using YOLO and the ZED

For a detailed explanation, please refer to the documentation: https://www.stereolabs.com/docs/yolo/

<p align="center">
<img src="https://docs.stereolabs.com/yolo/images/zed-yolo-3D.jpg" width="600">
</p>

This repository contains multiple samples demonstrating how to use YOLO models with the ZED camera, utilizing the highly optimized TensorRT library, as well as a Python sample that uses PyTorch and the official Ultralytics package for YOLOv8.

Other samples using OpenCV DNN or YOLOv5 with the TensorRT API in [C++](https://github.com/stereolabs/zed-sdk/tree/master/object%20detection/custom%20detector/cpp) or [PyTorch](https://github.com/stereolabs/zed-sdk/tree/master/object%20detection/custom%20detector/python) can be found in the main [ZED SDK repository](https://github.com/stereolabs/zed-sdk/tree/master/object%20detection/custom%20detector/cpp).

These samples are designed to run state-of-the-art object detection models using the highly optimized TensorRT framework. Images are captured with the ZED SDK to detect 2D bounding boxes using YOLO, and the ZED SDK then extracts 3D information (localization, 3D bounding boxes) and performs tracking.

## YOLO v5, v6, v7, v8, v9, v10, v11, v12 using TensorRT and C++

There are two main ways of running a YOLO ONNX model with the ZED and TensorRT:

1. **[Recommended]** Use the `OBJECT_DETECTION_MODEL::CUSTOM_YOLOLIKE_BOX_OBJECTS` mode in the ZED SDK API to natively load a YOLO ONNX model. The inference code is fully optimized and internally uses TensorRT. The output is directly available as 2D and 3D tracked objects. This is the easiest and most optimized solution for supported models. ONNX to TensorRT engine generation (optimized model) is automatically handled by the ZED SDK.

   The model format is inferred from the output tensor size. If a future model uses a similar output format to the supported models, it should work.

2. **[Advanced - for unsupported models]** Use external code for TensorRT model inference, then ingest the detected boxes into the ZED SDK. This approach is for advanced users, as it requires maintaining the inference code. It is suitable for models not supported by the previously mentioned method.


<p align="center">
  <img src="https://raw.githubusercontent.com/sunsmarterjie/yolov12/51901136772609c36df65cec1131d54b4f1a44df/assets/tradeoff.svg" width=90%> <br>
  YOLO Comparison in terms of latency-accuracy (left) and FLOPs-accuracy (right) trade-offs (from <a href="https://github.com/sunsmarterjie/yolov12">YOLOv12</a>)
</p>



### Exporting the model to ONNX (mandatory step)

Refer to the documentation: https://www.stereolabs.com/docs/yolo/ to export your YOLO model to ONNX for compatibility. Depending on the version, there may be multiple export methods, and not all are compatible or optimized.

You can use the default model trained on the COCO dataset (80 classes) provided by the framework maintainers, or a custom-trained model.

### C++ Version: Native Inference (recommended)

In the folder [cpp_tensorrt_yolo_onnx_native](./cpp_tensorrt_yolo_onnx_native), you will find a sample that runs an ONNX model exported from a YOLO architecture using the ZED with the C++ API.

### Python Version: Native Inference

In the folder [python_tensorrt_yolo_onnx_native](./python_tensorrt_yolo_onnx_native), you will find a sample that runs an ONNX model exported from a YOLO architecture using the ZED with the Python API.

### C++ Version: External Inference

In the folder [cpp_tensorrt_yolo_onnx](./cpp_tensorrt_yolo_onnx), you will find a sample that runs an ONNX model exported from a YOLO architecture using the ZED with the C++ API, leveraging external TensorRT inference code and a bounding box ingestion function.

## YOLOv8 PyTorch

In the folder [pytorch_yolov8](./pytorch_yolov8), you will find a sample that interfaces the PyTorch Ultralytics package with the ZED SDK in Python.

This sample demonstrates how to detect custom objects using the official PyTorch implementation of YOLOv8 from a ZED camera and ingest them into the ZED SDK to extract 3D information and track each object.
