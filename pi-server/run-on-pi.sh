#!/bin/bash

set -xe

DEST_DIR="$HOME/projects/clip-on-vision-assist-client"

cd $DEST_DIR/scripts

if ! cat $HOME/.profile | grep "FIRST_TIME_EXPORT"; then
  ./setup.sh
fi

cd $DEST_DIR
python3 ./client.py
