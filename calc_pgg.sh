#!/bin/bash

models=( "base" "expX" "spur" "neCL" "twistX" "nebCor" "cre10" "synCG" )

for i in "${models[@]}"
do
	python notebooks/UF23/skymaps.py $i 30
done
