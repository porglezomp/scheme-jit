#!/bin/sh

status=0
python3.7 -m unittest discover || status=1
python3.7 -m doctest *.py || status=1
exit $status
