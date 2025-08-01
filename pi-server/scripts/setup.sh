#!/bin/bash

set -xe

function is_installed()
{
    command -v $1 >/dev/null 2>&1
}

function export_local_bin_path()
{
    if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' > ~/.bashrc
        echo 'export PATH="$HOME/.local/bin:$PATH"' > ~/.profile
        source ~/.bashrc
    fi
}

function install_uv()
{
    curl -LsSf https://astral.sh/uv/install.sh | sh
}

function install_general_dependencies()
{
    sudo apt install libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6 libgdm-dev libdb4o-cil-dev libpcap-dev build-essential zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev liblzma-dev
}

function install_audio_dependencies()
{
    sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
    sudo apt-get install ffmpeg libav-tools
}
function install_vision_dependencies()
{
    sudo apt-get install libatlas-base-dev libjasper-dev libhdf5-dev
}

export_local_bin_path

if ! is_installed uv; then
    install_uv
fi

install_general_dependencies
install_audio_dependencies
install_vision_dependencies
