requirements:
	pip install -r requirements.txt

lint:
	python -m flake8

test:
	pytest --cov notanorm -v tests

publish:
	rm -rf dist
	python3 setup.py bdist_wheel
	twine upload dist/*
