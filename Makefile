requirements:
	pip install -r requirements.txt

requirements-pymysql:
	pip install -r requirements-pymysql.txt

lint:
	python -m flake8
	python -m black --check .

test:
	pytest -n 2 --cov notanorm -v tests --db sqlite --db jsondb

test-all:
	pytest --cov notanorm -v tests --db mysql --db sqlite --db jsondb

test-mysql:
	pytest -v tests --cov notanorm --cov-append --db mysql

publish:
	rm -rf dist
	python3 setup.py bdist_wheel
	twine upload dist/*

install-hooks:
	pre-commit install
