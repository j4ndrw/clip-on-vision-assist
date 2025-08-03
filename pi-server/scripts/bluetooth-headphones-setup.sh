#!/bin/bash

set -xe

DEVICE_MAC=$1

{
    echo "power on"
    echo "agent on"
    echo "scan on"

    while true; do
        if ! bluetoothctl paired-devices | grep -q "$DEVICE_MAC"; then
            echo "pair $DEVICE_MAC"
        fi

        if bluetoothctl info "$DEVICE_MAC" | grep -q "Connected: yes"; then
            echo "trust $DEVICE_MAC"
            echo "scan off"
            echo "exit"

            sleep 3

            AUDIO_CARD=$(pactl list cards | grep 'Name:' | awk -F': ' '{print $2}')
            pactl set-card-profile $AUDIO_CARD headset_head_unit
            exit 0
            break
        else
            echo "connect $DEVICE_MAC"
            sleep 1
        fi
    done
} | bluetoothctl
