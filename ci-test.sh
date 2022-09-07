#!/bin/bash
set -e

make requirements

make lint

. ./env/bin/activate || . ./env/Scripts/activate
make test

# this tests pymysql
. ./env/bin/activate || . ./env/Scripts/activate
make test-mysql
deactivate

python3 -m virtualenv env-pymysql
. ./env-pymysql/bin/activate || . ./env-pymysql/Scripts/activate
make requirements-pymysql
make test-mysql
