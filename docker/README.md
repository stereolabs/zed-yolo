# Docker

### Prerequisite

Setup Docker and nvidia-docker, see https://github.com/stereolabs/zed-docker

### Build the image

		docker build -t zed-yolo .

### Run the image

Replace "/<some_path_to_svo_files>" with an actual local absolute path containing SVO files :

		xhost +si:localuser:root
		nvidia-docker run -it --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --env QT_X11_NO_MITSHM=1 --mount type=bind,source=/<some_path_to_svo_files>,target=/data,readonly zed-yolo
