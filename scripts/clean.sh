#!/bin/sh
cd ../
rm -rf build/
rm -rf bin/
rm CMakeCache.txt
find . -name "cmake_install.cmake" -delete
find . -type d -name "CMakeFiles" -exec rm -rf {} +
