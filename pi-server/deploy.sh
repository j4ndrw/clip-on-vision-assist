#!/bin/bash

set -xe

USER=$1
IP=$2

DEST_DIR="/home/$USER/projects/clip-on-vision-assist-client"

function remote_cmd()
{
    ssh $USER@$IP $1
}

rsync \
    --rsync-path="mkdir -p $DEST_DIR && rsync" \
    -rv --progress ./ $USER@$IP:$DEST_DIR

remote_cmd "\
    chmod +x $DEST_DIR/run.sh && \
    chmod +x $DEST_DIR/scripts/* && \
    sudo cp $DEST_DIR/run.sh /usr/local/bin/clip-on-vision-assist-client && \
    \
    $DEST_DIR/scripts/setup.sh && \
    \
    mkdir -p /home/$USER/.config/systemd/user && \
    sudo cp $DEST_DIR/firmware/user/.config/systemd/user/clip-on-vision-assist-client.service /home/$USER/.config/systemd/user/clip-on-vision-assist-client.service && \
    sudo chmod 640 /home/$USER/.config/systemd/user/clip-on-vision-assist-client.service && \
    \
    sudo ln -sf /usr/lib/systemd/user/default.target /home/$USER/.config/systemd/user/default.target && \
    \
    sudo systemctl daemon-reload && \
    systemctl --user enable clip-on-vision-assist-client && \
    sudo reboot now
"
