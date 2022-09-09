#!/bin/bash
set -e

# this tests sqlite, mysqlclient
cp ci-my.cnf ~/.my.cnf

python3 -m virtualenv env
make requirements

. ./env/bin/activate || . ./env/Scripts/activate

make lint
make test


# test mysqlclient
make test-mysql


deactivate
python3 -m virtualenv env-pymysql
. ./env-pymysql/bin/activate || . ./env-pymysql/Scripts/activate

# test pymysql
make requirements-pymysql
make test-mysql
