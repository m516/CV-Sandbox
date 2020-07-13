#!/bin/sh

sudo apt-get install git make gcc

git submodule update --init --recursive

cd ~
mkdir tools
cd tools

git clone https://github.com/Kitware/CMake.git
cd CMake
./bootstrap && make && sudo make install

#GLEW dependencies
sudo apt-get install libxinerama-dev libxcursor-dev
#GLEW and GLFW
sudo apt-get install libglfw3-dev libglew-dev
