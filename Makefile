SHELL := /bin/bash

init:
	python3 -m venv .venv; \
	touch .env; \
	source .venv/bin/activate; \
	python3 -m pip install -U pip; \
	python3 -m pip install -U -r requirements.txt; \
