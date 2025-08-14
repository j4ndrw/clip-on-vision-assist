#!/bin/bash

set -xe

function build_frontend()
{
    cd ./frontend
    rm -rf ./dist
    npm i
    npm run build
    cd ..
}

build_frontend
