#!/bin/bash


for file in {EXPONENT1-50,PLATFORM1-50,PLATFORM2-50}; do
	python3 ./script/data_pca.py \
		--data ./data/$file".normalized_l2.data.tsv" \
		--meta ./data/$file".normalized_l2.meta.tsv"
done
