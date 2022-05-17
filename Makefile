.DEFAULT_GOAL := all
isort = isort deepdrivemd test
black = black --target-version py37 deepdrivemd test

.PHONY: venv
venv:
	python -m venv .venv

.PHONY: install-dev
install-dev:
	python -m pip install --upgrade wheel pip
	python -m pip install -r requirements_dev.txt

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	$(black) --check --diff
	flake8 deepdrivemd/ test/
	pydocstyle deepdrivemd/

.PHONY: mypy
mypy:
	mypy --config-file setup.cfg --package deepdrivemd
	# mypy --config-file setup.cfg test/

.PHONY: all
all: format lint mypy
