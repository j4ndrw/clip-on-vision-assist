#!/bin/bash

set -xe

function freeze_backend_packages()
{
    uv export --no-hashes --no-header --no-annotate --no-dev --format requirements.txt > requirements.txt
}

function build_frontend()
{
    cd ./frontend
    rm -rf ./dist
    npm i
    npm run build
    cd ..
}

freeze_backend_packages
build_frontend
