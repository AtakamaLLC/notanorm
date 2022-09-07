requirements:
	python3 -mpip install -r requirements.txt

requirements-pymysql:
	python3 -mpip install -r requirements-pymysql.txt

lint:
	python3 -m flake8

test:
	python3 -mpytest -n 2 --cov notanorm -v tests

test-all:
	python3 -mpytest --cov notanorm -v tests --db mysql --db sqlite

test-mysql:
	python3 -mpytest -v tests --cov notanorm --cov-append --db mysql

publish:
	rm -rf dist
	python3 setup.py bdist_wheel
	python3 -mtwine upload dist/*
