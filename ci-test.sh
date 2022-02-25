#!/bin/bash
set -e

# this tests sqlite, mysqlclient
cp ci-my.cnf ~/.my.cnf

python3 -m virtualenv env
. ./env/bin/activate || . ./env/Scripts/activate
make requirements
make lint
make test-all

deactivate

# this tests pymysql
python3 -m virtualenv env-pymysql
. ./env-pymysql/bin/activate || . ./env-pymysql/Scripts/activate
make requirements-pymysql
make test-mysql
