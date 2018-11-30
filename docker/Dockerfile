FROM stereolabs/zed:ubuntu1604-cuda9.0-zed2.7-gl

ENV CUDNN_VERSION 7.4.1.5

RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-dev python3-pip unzip sudo libopencv-dev -y
RUN apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda9.0 libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && apt-mark hold libcudnn7

RUN sudo chmod 777 -R /usr/local/zed

RUN pip3 install --upgrade pip opencv-python
RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker
WORKDIR /home/docker

RUN git clone http://github.com/stereolabs/zed-python.git
RUN cd zed-python; sudo pip3 install -r requirements.txt; sudo python3 setup.py install

RUN git clone https://github.com/stereolabs/zed-yolo.git
RUN cd zed-yolo/libdarknet ; make -j8

#RUN cd /home/docker/zed-yolo/zed_cpp_sample/ ; mkdir build ; cd build; cmake .. ; make

WORKDIR /home/docker/zed-yolo/
RUN wget https://pjreddie.com/media/files/yolov3-tiny.weights
RUN ln -s /home/docker/zed-yolo/yolov3-tiny.weights /home/docker/zed-yolo/zed_python_sample/
CMD /bin/bash
