#!/bin/sh

status=0
mypy --strict ./*.py || status=1
pycodestyle ./*.py || status=1
# for some reason Travis-CI isort doesn't know about dataclasses???
isort --diff --check-only --recursive -b dataclasses || status=1
exit $status
