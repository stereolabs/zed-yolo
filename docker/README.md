# Docker

## Prerequisite

Setup Docker and nvidia-docker, see https://github.com/stereolabs/zed-docker

## Build the image

		docker build -t zed-yolo .

## Run the image

Following the [instruction given here](https://github.com/stereolabs/zed-docker#opengl-support):

```Bash
xhost +si:localuser:root
```

Run the image :

```Bash
nvidia-docker run -it --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source=/<some_path_to_svo_files>,target=/data,readonly --env QT_X11_NO_MITSHM=1 zed-yolo
```

### Run the python sample

From within the container start python the sample :

```Bash
cd zed_python_sample/
python3 darknet_zed.py -s /data/<path/to/SVO>
```

### Run the C++ sample

From within the container compile the C++ sample :

```Bash
cd /home/docker/zed-yolo/zed_cpp_sample/ ; mkdir build ; cd build; cmake .. ; make ; cd /home/docker/zed-yolo/
```

Then start it :

```Bash
zed_cpp_sample/build/darknet_zed libdarknet/data/coco.names libdarknet/cfg/yolov3-tiny.cfg yolov3-tiny.weights /data/<path/to/SVO>
```