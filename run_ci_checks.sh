#!/bin/bash
./run_autoformat.sh
mypy src tests experiments data
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc
pytest tests/
