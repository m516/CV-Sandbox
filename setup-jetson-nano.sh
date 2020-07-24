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
sudo apt-get install -y libssl-dev
git clone https://github.com/Kitware/CMake.git
cd CMake
./bootstrap
make -j5
sudo make install

# CUDA is installed by default on Jetsons, but it isn't in the path.
# Do so manually
printf "\n\n#Add CUDA to path\nexport PATH=/usr/local/cuda/bin:\$PATH\n" >> ~/.bashrc



