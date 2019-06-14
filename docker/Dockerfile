FROM stereolabs/zed:ubuntu1604-cuda9.0-zed2.8-gl

ENV CUDNN_VERSION 7.4.1.5

RUN apt-get update && apt-get upgrade -y ; \
     apt-get install python3-dev python3-pip unzip sudo libopencv-dev apt-transport-https ca-certificates gnupg software-properties-common wget -y ; \
     apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda9.0 libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && apt-mark hold libcudnn7

RUN sudo chmod 777 -R /usr/local/zed ; \
    pip3 install --upgrade pip opencv-python ; \
    echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker
WORKDIR /home/docker

RUN git clone http://github.com/stereolabs/zed-python.git ; \
    cd zed-python; sudo pip3 install -r requirements.txt; sudo python3 setup.py install ; cd .. ; sudo rm -rf zed-python

RUN git clone https://github.com/stereolabs/zed-yolo.git ; \
    cd zed-yolo ; sudo bash cmake_apt_update.sh ; \
	cd libdarknet ; make -j8

RUN git clone https://github.com/alexeyAB/darknet.git ; \
	cd darknet ; cmake . ; make -j8

WORKDIR /home/docker/zed-yolo/
RUN wget https://pjreddie.com/media/files/yolov3-tiny.weights ; \ 
	ln -s /home/docker/zed-yolo/yolov3-tiny.weights /home/docker/zed-yolo/zed_python_sample/ ; \
    ln -s /home/docker/zed-yolo/yolov3-tiny.weights /home/docker/darknet/
CMD /bin/bash
