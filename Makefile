requirements:
	pip install -r requirements.txt

requirements-pymysql:
	pip install -r requirements-pymysql.txt

lint:
	python -m flake8

test:
	pytest --cov notanorm -v tests

test-all:
	pytest --cov notanorm -v tests --db mysql --db sqlite

test-mysql:
	pytest -v tests --cov notanorm --cov-append --db mysql

publish:
	rm -rf dist
	python3 setup.py bdist_wheel
	twine upload dist/*
