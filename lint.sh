#!/bin/sh

status=0
mypy --strict *.py || status=1
pycodestyle *.py || status=1
exit $status
