#!/bin/bash

set -xe

DEST_DIR="$HOME/projects/clip-on-vision-assist-client"

cd $DEST_DIR/scripts

if ! cat $HOME/.profile | grep "FIRST_TIME_EXPORT"; then
  ./setup.sh
fi
./bluetooth-headphones-setup.sh

cd $DEST_DIR
/usr/bin/python3 ./client.py
