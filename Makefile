requirements:
	pip install -r requirements.txt

lint:
	python -m flake8

test:
	pytest -v tests

publish:
	python3 setup.py bdist_wheel
	twine upload dist/*
