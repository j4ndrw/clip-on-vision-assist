#!/bin/bash

set -xe

USER=$1
IP=$2

DEST_DIR="/home/$USER/projects/clip-on-vision-assist-satellite"

function remote_cmd()
{
    ssh $USER@$IP $1
}

rsync --rsync-path="mkdir -p $DEST_DIR && rsync" -rv --progress ./ $USER@$IP:$DEST_DIR
remote_cmd "sudo chmod +x $DEST_DIR/post-deploy.sh && $DEST_DIR/post-deploy.sh"
