# Docker

## Prerequisite

Setup Docker and nvidia-docker, see https://github.com/stereolabs/zed-docker

## Build the image


		docker build -t zed-yolo .

## Run the image

		nvidia-docker run -it --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --env QT_X11_NO_MITSHM=1 zed-yolo