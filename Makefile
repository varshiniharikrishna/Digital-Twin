.PHONY: all test
all:
	python3 run_all.py

test:
	pytest -q
