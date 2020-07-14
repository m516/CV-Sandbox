# Setup <!-- omit in toc -->
## Introduction <!-- omit in toc -->

This document describes my setup as of 6/21/2020.
Note that links, procedures, and my setup may change significantly and I may not update this page often.

If this document is outdated, please don't heasitate to create a new [issue](https://github.com/m516/CV-Sandbox/issues).

------------------------------

## Table of Contents <!-- omit in toc -->
- [My Personal Setup](#My-Personal-Setup)
  - [Windows 10](#Windows-10)
  - [Ubuntu](#Ubuntu)
- [Computer Vision Libraries I Use](#Computer-Vision-Libraries-I-Use)
- [Installing OpenCV via `vcpkg`](#Installing-OpenCV-via-vcpkg)

------------------------------

## My Personal Setup
### Windows 10
Below is a list of all software I'm using on a Windows 10 device to build and run the CV projects in
this repository.
All the software I use right now is free.

* [Visual Studio](https://visualstudio.microsoft.com/), though [Visual Studio Code](https://code.visualstudio.com/) works as well.
* MSVC as a C++ compiler (integrated with Visual Studio). [GCC](https://gcc.gnu.org/), G++, and [Clang](https://clang.llvm.org/)
  are viable alternatives. MSVC is integrated into Visual Studio.
* [CMake](https://cmake.org/) is a cross-platform build tool, and is integrated into Visual Studio.
* [vcpkg](https://docs.microsoft.com/en-us/cpp/build/vcpkg?view=vs-2019) is a library package manager.
* [Git](https://git-scm.com/) is a great version control system, and I use it extensively for other projects. 
  It's integrated in Visual Studio and VS Code, but I prefer to use the command line or 
  [Github Desktop](https://desktop.github.com/) when dealing with version control.

### Ubuntu
Below is a list of all software I'm using on an NVidia Jetson Nano to build and run the same CV in this repository.
All the software I use right now is free.


* [Visual Studio](https://visualstudio.microsoft.com/), though [Visual Studio Code](https://code.visualstudio.com/) works as well.
* [GCC and G++](https://gcc.gnu.org/) as a C/C++ compiler. MSVC and [Clang](https://clang.llvm.org/) are viable alternatives.
* [CMake](https://cmake.org/) is a cross-platform build tool. It can be installed with `apt`, but the `apt` version as of 7/13/2020 is much older than the current version, and this project isn't compatible with the `apt` version of CMake yet.
* [vcpkg](https://docs.microsoft.com/en-us/cpp/build/vcpkg?view=vs-2019) is a library package manager. It prefers x64 and x86 architectures because of the CMake and Ninja binaries it installs, but a simple workaround has been found for the Nano's architecture. See https://github.com/microsoft/vcpkg/issues/10955 and [my setup script](../setup.sh).
* [Git](https://git-scm.com/) is a great version control system, and I use it extensively for other projects. 
  It's integrated in Visual Studio and VS Code, but I prefer to use the command line in Ubuntu.

## Computer Vision Libraries I Use
* [OpenCV](https://opencv.org/) is an extensive, fast open-source computer vision library. 
It can be installed in [many different ways](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html).
I personally use `sudo apt-get install libopencv-dev` to install OpenCV on Ubuntu, but vcpkg is an
alternate
See ["Installing OpenCV via `vcpkg`"](#installing-opencv-via-vcpkg) for more info about this processes.

## Installing OpenCV via `vcpkg`
[vcpkg](https://docs.microsoft.com/en-us/cpp/build/vcpkg?view=vs-2019) is an intuitive, cross-platform 
library package manager for C and C++, and it takes minimal effort to integrate with  Visual Studio, VS Code, 
and any CMake setup.

If you have vcpkg installed, run the following command:
```bash
vcpkg install opencv4:x64-windows
```
and vcpkg will install the 64-bit version of OpenCV.

If you want to learn more about vcpkg, here are the links that helped me use it:
* [Microsoft's official documentation](https://docs.microsoft.com/en-us/cpp/build/vcpkg?view=vs-2019)
* [Github repository for vcpkg](https://github.com/Microsoft/vcpkg)
