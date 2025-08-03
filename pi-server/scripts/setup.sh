#!/bin/bash

set -xe

function first_time_exports()
{
    echo 'export FIRST_TIME_EXPORT="done"' >> ~/.bashrc
    echo 'export FIRST_TIME_EXPORT="done"' >> ~/.profile
    source ~/.bashrc
}

function install_general_dependencies()
{
    sudo apt install -y \
        libssl-dev \
        libncurses5-dev \
        libsqlite3-dev \
        libreadline-dev \
        libtk8.6 \
        libgdm-dev \
        libdb4o-cil-dev \
        libpcap-dev \
        build-essential \
        zlib1g-dev \
        libffi-dev \
        libssl-dev \
        libbz2-dev \
        libreadline-dev \
        liblzma-dev
}

function install_audio_dependencies()
{
    sudo apt-get install -y \
        libasound-dev \
        portaudio19-dev \
        libportaudio2 \
        libportaudiocpp0 \
        ffmpeg
}

function install_vision_dependencies()
{
    sudo apt-get install -y \
        libcap-dev \
        libavformat-dev \
        libavdevice-dev \
        libavcodec-dev \
        libavfilter-dev \
        libavutil-dev \
        libatlas-base-dev \
        ffmpeg \
        libopenjp2-7 \
        libcamera-dev \
        libkms++-dev \
        libfmt-dev \
        libdrm-dev
}


if [ -z "${FIRST_TIME_EXPORT}" ]; then
    first_time_exports
    install_general_dependencies
    install_audio_dependencies
    install_vision_dependencies
fi
