#!/bin/bash
declare -a func=(6 7 8 9) 
for j in {0..3}
do
	python3 ndbjde.py -acc 0.001 -a 1 -flag 0 -r 10 -p 150 -f "${func[$j]}"
done
