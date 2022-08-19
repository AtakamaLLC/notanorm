#!/bin/bash
set -e

# this tests sqlite, mysqlclient
#cp ci-my.cnf ~/.my.cnf

python3 -m virtualenv env
make requirements

# lint + sqlite + mysql in parallel
make lint &

(
. ./env/bin/activate || . ./env/Scripts/activate
make test
) &


(
# this tests pymysql
. ./env/bin/activate || . ./env/Scripts/activate
make test-mysql
deactivate

python3 -m virtualenv env-pymysql
. ./env-pymysql/bin/activate || . ./env-pymysql/Scripts/activate
make requirements-pymysql
make test-mysql

) &

wait
