#!/bin/sh


dims_test_dir="output/dims_test"
mkdir -p $dims_test_dir
for file in {EXPONENT1-50,PLATFORM1-50,PLATFORM2-50}; do
	mkdir -p $dims_test_dir/$file
	for dimred in {pca,lda,lsdr}; do
		mkdir -p $dims_test_dir/$file/$dimred
		#for model in {gnb,lr,lda,svm_lin,svm_rbf}; do
		for model in {svm_lin,svm_rbf}; do
			mkdir -p $dims_test_dir/$file/$dimred/$model
			for ndims in {1..30}; do
				python3 ./script/t1_methods.py \
					--data ./data/$file".normalized_l2.data.tsv" \
					--meta ./data/$file".normalized_l2.meta.tsv" \
					--classifier $model \
					--dim-reduc $dimred --reduce-dim-to $ndims \
					--output-txt-dir $dims_test_dir/$file/$dimred/$model \
					--output-png-dir $dims_test_dir/$file/$dimred/$model
			done > $dims_test_dir/$file/$dimred/$model/log
		done
	done
done
