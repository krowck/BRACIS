#!/bin/bash
declare -a func=(14 15 16)
for j in {0..2}
do
	python3 ndbjde.py -acc 0.001 -a 1 -flag 0 -r 5 -p 150 -f "${func[$j]}"
done
