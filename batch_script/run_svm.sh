#!/bin/bash


for model in {svm_lin,svm_rbf}; do
	for file in {EXPONENT1-50,PLATFORM1-50,PLATFORM2-50}; do
		python3 ./script/t1_methods.py \
			--data ./data/$file".normalized_l2.data.tsv" \
			--meta ./data/$file".normalized_l2.meta.tsv" \
			--classifier $model
		for dimred in {pca,lda}; do
		#for dimred in {pca,lda,lsdr}; do
			for ndims in {20,26,40}; do
				python3 ./script/t1_methods.py \
					--data ./data/$file".normalized_l2.data.tsv" \
					--meta ./data/$file".normalized_l2.meta.tsv" \
					--classifier $model \
					--dim-reduc $dimred --reduce-dim-to $ndims
			done
		done
	done
done
