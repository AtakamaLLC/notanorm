requirements:
	pip install -r requirements.txt

lint:
	python -m flake8

test:
	pytest --cov notanorm -v tests

test-all:
	pytest --cov notanorm -v tests --db mysql --db sqlite

publish:
	rm -rf dist
	python3 setup.py bdist_wheel
	twine upload dist/*
