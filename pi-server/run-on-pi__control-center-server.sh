#!/bin/bash

set -xe

DEST_DIR="$HOME/projects/clip-on-vision-assist-satellite"

cd $DEST_DIR
python3 -m gunicorn control_center_server:app --bind 0.0.0.0:42068
