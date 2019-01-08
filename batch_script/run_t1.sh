#!/bin/bash


for model in {gnb,lr,lda}; do
	for file in {EXPONENT1-50,PLATFORM1-50,PLATFORM2-50}; do
		for pca in {none,20,26,40}; do
			python3 ./script/t1_methods.py \
				--data ./data/$file".normalized_l2.data.tsv" \
				--meta ./data/$file".normalized_l2.meta.tsv" \
				--model $model --pca $pca
		done
	done
done
