#!/bin/sh

sudo apt-get -y install git make gcc

git submodule update --init --recursive

cd ~
mkdir .tools
cd .tools

# OpenCV--CUDA accelerated
git clone https://github.com/mdegans/nano_build_opencv.git
cd nano_build_opencv
./build_opencv.sh

cd ~
cd .tools

# latest CMake
sudo apt-get purge cmake
git clone https://github.com/Kitware/CMake.git
cd CMake
./bootstrap && make && sudo make install

# vcpkg
#cd .tools/
#sudo apt-get -y install curl unzip tar ninja-buid
#git clone https://github.com/microsoft/vcpkg
#cd vcpkg
#./bootstrap-vcpkg.sh --useSystemBinaries -disableMetrics



