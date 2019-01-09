#!/bin/bash


for model in {gnb,lr,lda,svm_lin,svm_rbf}; do
	for file in {EXPONENT1-50,PLATFORM1-50,PLATFORM2-50}; do
		python3 ./script/t1_methods.py \
			--data ./data/$file".normalized_l2.data.tsv" \
			--meta ./data/$file".normalized_l2.meta.tsv" \
			--classifier $model \
			--dim-reduc lsdr --reduce-dim-to 26
			#--output-txt-dir ./develop/with_scale \
			#--output-png-dir ./develop/with_scale \
	done
done
