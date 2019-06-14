# Docker

## Prerequisite

Setup Docker and nvidia-docker, see https://github.com/stereolabs/zed-docker

## Build the image

```Bash
docker build -t zed-yolo .
```

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

From within the container the C++ sample can be started :

```Bash
cd ~/darknet
./uselib libdarknet/data/coco.names libdarknet/cfg/yolov3-tiny.cfg yolov3-tiny.weights /data/<path/to/SVO>
```
