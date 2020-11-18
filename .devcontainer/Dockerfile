# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.148.1/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version: debian-10, debian-9, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="buster"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

# Install additional packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends cmake make gcc g++ xorg-dev libopencv-dev

# TODO review over break. Is this a good place to build the project?
