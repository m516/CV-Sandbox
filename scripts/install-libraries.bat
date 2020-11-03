@echo off

Rem try to update the Git repository if possible
where /q vcpkg
IF ERRORLEVEL 1 (
    ECHO Couldn't find vcpkg. This is a library package manager used by this project.
    ECHO Check out https://docs.microsoft.com/en-us/cpp/build/vcpkg for more information about vcpkg
	exit
)

vcpkg integrate install
vcpkg remove --outdated

Rem try to install important packages
vcpkg install opencv4:x64-windows opencv4:x86-windows glad:x64-windows glad:x86-windows glew:x64-windows glew:x86-windows cuda:x86-windows cuda:x64-windows pthreads:x86-windows pthreads:x64-windows
IF ERRORLEVEL 1 (
    ECHO Library installation failed :(
	exit
)

ECHO Library installation complete. Yay!
pause