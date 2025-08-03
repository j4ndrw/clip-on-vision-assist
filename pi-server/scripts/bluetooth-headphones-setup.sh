#!/bin/bash

set -xe

function connect_bluetooth_headphones()
{
    DEVICE_MAC=$1

    {
        echo "power on"
        echo "agent on"
        echo "scan on"

        while true; do
            if bluetoothctl devices | grep -q "$DEVICE_MAC"; then
                echo "Device found: $DEVICE_MAC"
                break
            fi
            sleep 1
        done

        echo "scan off"

        if bluetoothctl paired-devices | grep -q "$DEVICE_MAC"; then
            echo "Device is already paired: $DEVICE_MAC"
        else
            echo "pair $DEVICE_MAC"

            while true; do
                if bluetoothctl paired-devices | grep -q "$DEVICE_MAC"; then
                    echo "Device paired: $DEVICE_MAC"
                    break
                fi
                sleep 1
            done
        fi

        while true; do
            if bluetoothctl paired-devices | grep -q "$DEVICE_MAC"; then
                echo "Device paired: $DEVICE_MAC"
                break
            fi
            sleep 1
        done

        echo "connect $DEVICE_MAC"

        while true; do
            if bluetoothctl info "$DEVICE_MAC" | grep -q "Connected: yes"; then
                echo "Device connected: $DEVICE_MAC"
                break
            fi
            sleep 1
        done

        echo "trust $DEVICE_MAC"
        echo "exit"
    } | bluetoothctl
}

connect_bluetooth_headphones $1

sleep 3

AUDIO_CARD=$(pactl list cards | grep 'Name:' | awk -F': ' '{print $2}')
pactl set-card-profile $AUDIO_CARD headset_head_unit
