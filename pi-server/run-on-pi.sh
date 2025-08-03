#!/bin/bash

set -xe

DEST_DIR="$HOME/projects/clip-on-vision-assist-client"

BLUETOOTH_HEADPHONES_MAC=
source <(grep = $DEST_DIR/.env)


cd $DEST_DIR/scripts

if ! cat $HOME/.profile | grep "FIRST_TIME_EXPORT"; then
  ./setup.sh
fi
./bluetooth-headphones-setup.sh $BLUETOOTH_HEADPHONES_MAC

pip install --user -r $DEST_DIR/requirements.txt

cd $DEST_DIR
python3 ./client.py
