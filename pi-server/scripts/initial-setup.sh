#!/bin/bash

set -xe

function install_general_dependencies()
{
    sudo apt install -y libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6 libgdm-dev libdb4o-cil-dev libpcap-dev build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev liblzma-dev
}

function install_audio_dependencies()
{
    sudo apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
    sudo apt-get install -y ffmpeg
}
function install_vision_dependencies()
{
    sudo apt-get install -y libcap-dev libavformat-dev libavdevice-dev libavcodec-dev libavfilter-dev libavutil-dev
    sudo apt install -y libatlas-base-dev ffmpeg libopenjp2-7
    sudo apt install -y libcamera-dev libkms++-dev libfmt-dev libdrm-dev
}

export_local_bin_path

install_general_dependencies
install_audio_dependencies
install_vision_dependencies
