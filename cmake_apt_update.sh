#!/usr/bin/env bash

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

# https://apt.kitware.com/
if [  -n "$(uname -a | grep Ubuntu)" ]; then
  apt-get update ; apt-get install lsb-release apt-transport-https ca-certificates gnupg software-properties-common wget -y
  wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
  codename=`lsb_release --codename | cut -f2`
  apt-add-repository 'deb https://apt.kitware.com/ubuntu/ '$codename' main'
  apt-get install kitware-archive-keyring
  apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA
  apt-get cmake
else
  echo "This script is only meant for Ubuntu"
fi  