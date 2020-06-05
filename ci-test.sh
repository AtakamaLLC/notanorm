#!/bin/bash

python3 -m virtualenv env
. ./env/bin/activate || . ./env/Scripts/activate
make requirements
make lint
make test
