requirements:
	pip install -r requirements.txt

lint:
	python -m flake8

test:
	pytest -v tests
