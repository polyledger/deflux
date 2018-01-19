.PHONY: clean-pyc clean-build

clean-pyc:
	find . -maxdepth 1 -name '*.pyc' -exec rm -f {} +
	find . -maxdepth 1  -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

lint:
	flake8 cvar.py deflux.py tests.py

test:
	python tests.py

help:
	@echo "clean-pyc"
	@echo "    Remove python artifacts."
	@echo "lint"
	@echo "    Check style with flake8."
	@echo "test"
	@echo "    Run py.test"
