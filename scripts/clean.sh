#!/bin/sh

git submodule foreach git clean -xdf
git submodule foreach git reset --hard origin/master
rm -rf build/
rm -rf bin/
rm CMakeCache.txt
find . -name "cmake_install.cmake" -delete
find . -type d -name "CMakeFiles" -exec rm -rf {} +
