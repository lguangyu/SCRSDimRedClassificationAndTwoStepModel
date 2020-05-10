#!/bin/bash


data="./data/COMBINED.normalized_l2.data.tsv"
meta="./data/COMBINED.phase.normalized_l2.meta.tsv"
#COMBINED.strain.normalized_l2.meta.tsv
for model in {gnb,lr,lda,svm_lin,svm_rbf}; do
	python3 ./script/t1_methods.py \
		--data $data \
		--meta $meta \
		--classifier $model
	for dimred in {pca,lda,lsdr}; do
		for ndims in {20,26,40}; do
			python3 ./script/t1_methods.py \
				--data $data \
				--meta $meta \
				--classifier $model \
				--dim-reduc $dimred --reduce-dim-to $ndims
		done
	done
done
