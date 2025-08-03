#!/bin/bash

set -xe

DEST_DIR="$HOME/projects/clip-on-vision-assist-client"
FIRMWARE_DIR="$DEST_DIR/firmware"

HOTSPOT_SSID=
HOTSPOT_PASSWORD=

source <(grep = $DEST_DIR/.env)

function setup_static_ip()
{
    sudo cp $FIRMWARE_DIR/etc/dhcpcd.conf /etc/dhcpcd.conf
    sudo service dhcpcd restart
}

function configure_hostapd()
{
    sed -i "s/<SSID>/$HOTSPOT_SSID/g" $FIRMWARE_DIR/etc/hostapd/hostapd.conf
    sed -i "s/<PASSWORD>/$HOTSPOT_PASSWORD/g" $FIRMWARE_DIR/etc/hostapd/hostapd.conf

    sudo cp $FIRMWARE_DIR/etc/hostapd/hostapd.conf /etc/hostapd/hostapd.conf

    if ! sudo cat /etc/default/hostapd | grep "DAEMON_CONF=\"/etc/hostapd/hostapd.conf\""; then
        sudo echo "DAEMON_CONF=\"/etc/hostapd/hostapd.conf\"" | sudo tee -a /etc/default/hostapd
    fi
}

function configure_dnsmasq()
{
    if ! sudo ls /etc/dnsmasq.conf.bak | grep "/etc/dnsmasq.conf.bak"; then
        sudo mv /etc/dnsmasq.conf /etc/dnsmasq.conf.bak
    fi
    sudo cp $FIRMWARE_DIR/etc/dnsmasq.conf /etc/dnsmasq.conf
}

sudo systemctl stop hostapd
sudo systemctl stop dnsmasq


setup_static_ip
configure_hostapd
configure_dnsmasq

sudo systemctl unmask hostapd
sudo systemctl enable hostapd
sudo systemctl start hostapd
sudo systemctl start dnsmasq
