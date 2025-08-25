#!/bin/bash

set -xe

DEST_DIR="$HOME/projects/clip-on-vision-assist-satellite"
SCRIPTS_DIR="$DEST_DIR/scripts"

CLIENT_LINKED_EXECUTABLE="/usr/local/bin/clip-on-vision-assist-client"
CONTROL_CENTER_SERVER_LINKED_EXECUTABLE="/usr/local/bin/clip-on-vision-assist-control-center-server"

SYSTEMD_COVA_CLIENT_SERVICE_NAME="clip-on-vision-assist-client"
SYSTEMD_COVA_CLIENT_SERVICE="$SYSTEMD_COVA_CLIENT_SERVICE_NAME.service"
SYSTEMD_COVA_CONTROL_CENTER_SERVER_SERVICE_NAME="clip-on-vision-assist-control-center-server"
SYSTEMD_COVA_CONTROL_CENTER_SERVER_SERVICE="$SYSTEMD_COVA_CONTROL_CENTER_SERVER_SERVICE_NAME.service"
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"

SYSTEMD_CREATE_HOTSPOT_SERVICE_NAME="create-hotspot-ap"
SYSTEMD_CREATE_HOTSPOT_SERVICE="$SYSTEMD_CREATE_HOTSPOT_SERVICE_NAME.service"
ROOT_SYSTEMD_DIR="/etc/systemd/system"

function enable_systemd_services()
{
    sudo systemctl daemon-reload
    systemctl --user enable $SYSTEMD_COVA_CLIENT_SERVICE
    systemctl --user enable $SYSTEMD_COVA_CONTROL_CENTER_SERVER_SERVICE
    sudo systemctl enable $SYSTEMD_CREATE_HOTSPOT_SERVICE
}

function allow_executables()
{
    chmod +x $DEST_DIR/run-on-pi__client.sh
    chmod +x $DEST_DIR/run-on-pi__control-center-server.sh
    chmod +x $SCRIPTS_DIR/*
}

sed -i "s#<DEST_DIR>#$DEST_DIR#g" $DEST_DIR/scripts/setup-hotspot.sh
sed -i "s#<DEST_DIR>#$DEST_DIR#g" $DEST_DIR/firmware/etc/systemd/system/$SYSTEMD_CREATE_HOTSPOT_SERVICE

allow_executables
sudo ln -sf $DEST_DIR/run-on-pi__client.sh $CLIENT_LINKED_EXECUTABLE
sudo ln -sf $DEST_DIR/run-on-pi__control-center-server.sh $CONTROL_CENTER_SERVER_LINKED_EXECUTABLE

$SCRIPTS_DIR/setup.sh

mkdir -p $USER_SYSTEMD_DIR

sudo cp $DEST_DIR/firmware/user/.config/systemd/user/$SYSTEMD_COVA_CLIENT_SERVICE $USER_SYSTEMD_DIR/$SYSTEMD_COVA_CLIENT_SERVICE
sudo chmod 640 $USER_SYSTEMD_DIR/$SYSTEMD_COVA_CLIENT_SERVICE

sudo cp $DEST_DIR/firmware/user/.config/systemd/user/$SYSTEMD_COVA_CONTROL_CENTER_SERVER_SERVICE $USER_SYSTEMD_DIR/$SYSTEMD_COVA_CONTROL_CENTER_SERVER_SERVICE
sudo chmod 640 $USER_SYSTEMD_DIR/$SYSTEMD_COVA_CONTROL_CENTER_SERVER_SERVICE

sudo chown $USER:$USER -R $USER_SYSTEMD_DIR
sudo ln -sf /usr/lib/systemd/user/default.target $USER_SYSTEMD_DIR/default.target

sudo cp $DEST_DIR/firmware/etc/systemd/system/$SYSTEMD_CREATE_HOTSPOT_SERVICE $ROOT_SYSTEMD_DIR/$SYSTEMD_CREATE_HOTSPOT_SERVICE

enable_systemd_services

pip install --user -r $DEST_DIR/requirements.txt

sudo reboot now
