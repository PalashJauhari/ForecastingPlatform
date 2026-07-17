.PHONY: help install test start kill ci

help:
	@echo "Targets: install | test | start | kill | ci"

install:
	pip install -r requirements.txt

test:
	pytest

start:
	./start.sh

kill:
	./kill.sh

ci: test
