#!/bin/bash
declare -a func=(19 20)
for j in {0..1}
do
	python3 ndbjde.py -acc 0.001 -a 1 -flag 0 -r 5 -p 150 -f "${func[$j]}"
done
