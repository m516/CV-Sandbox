# Setup <!-- omit in toc -->
## Introduction <!-- omit in toc -->

This document describes my setup as of 6/21/2020.
Note that links, procedures, and my setup may change significantly and I may not update this page often.

If this document is outdated, please don't heasitate to create a new [issue](https://github.com/m516/CV-Sandbox/issues).

------------------------------

## Table of Contents <!-- omit in toc -->
- [My Personal Setup](#my-personal-setup)
- [Computer Vision Libraries I Use](#computer-vision-libraries-i-use)
- [Installing OpenCV via `vcpkg`](#installing-opencv-via-vcpkg)

------------------------------

## My Personal Setup
This is the software I'm using on a Windows 10 device to build and run the CV projects I write in C++.
All the software I use right now is free.

* [Visual Studio](https://visualstudio.microsoft.com/), though [Visual Studio Code](https://code.visualstudio.com/) works as well.
* MSVC as a C++ compiler (integrated with Visual Studio). [GCC](https://gcc.gnu.org/), G++, and [Clang](https://clang.llvm.org/)
  are viable alternatives. MSVC is integrated into Visual Studio.
* [CMake](https://cmake.org/) is a cross-platform build tool, and is integrated into Visual Studio.
* [vcpkg](https://docs.microsoft.com/en-us/cpp/build/vcpkg?view=vs-2019) is a library package manager.
* [Git](https://git-scm.com/) is a great version control system, and I use it extensively for other projects. 
  It's integrated in Visual Studio and VS Code, but I prefer to use the command line or 
  [Github Desktop](https://desktop.github.com/) when dealing with version control.

## Computer Vision Libraries I Use
* [OpenCV](https://opencv.org/) is an extensive, fast open-source computer vision library. 
It can be installed in [many different ways](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html).
I personally use vcpkg to install OpenCV on my Windows 10 device, but vcpkg works on Mac OS and Linux with VS Code too. 
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
