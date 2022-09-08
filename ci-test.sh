#!/bin/bash
set -e

make requirements

make lint

python3 -mvenv env

. ./env/bin/activate || . ./env/Scripts/activate
make test

# test mysqlclient
make test-mysql
deactivate

# test pymysql
python3 -mvenv env-pymysql
. ./env-pymysql/bin/activate || . ./env-pymysql/Scripts/activate
make requirements-pymysql
make test-mysql
