#!/bin/bash
declare -a func=(10 11 12 13)
for j in {0..3}
do
	python3 ndbjde.py -acc 0.001 -a 1 -flag 0 -r 10 -p 150 -f "${func[$j]}"
done
