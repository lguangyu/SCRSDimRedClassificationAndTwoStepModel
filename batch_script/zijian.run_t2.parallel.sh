#!/bin/bash

mkdir -p .log # ouput log directory
mkdir -p output
mkdir -p output/zijian
mkdir -p output/zijian/t2


dataset="zijian-duo-label"
for cls1 in {gnb,lda,lr,svm_lin,svm_rbf,svm_lin_cv,svm_rbf_cv}; do
	for dr1 in {none,lda,lsdr,pca}; do
		for cls2 in {gnb,lda,lr,svm_lin,svm_rbf,svm_lin_cv,svm_rbf_cv}; do
			for dr2 in {none,lda,lsdr,pca}; do
				job_desc="t2.$cls1.$dr1.$cls2.$dr2"
					sbatch -J $job_desc \
					-o ".log/"$job_desc".log" \
					-e ".log/"$job_desc".err" \
					-p general -N1 -c8 --mem 8G \
					--wrap \
"# run experiments #
echo \$SLURM_JOB_ID >&2
module load python/3.7.1
. /home/li.gua/python-virtual-env/python-3.7.1-metagenomic-general/bin/activate

python3 ./script/t2_2level.py \\
	--dataset $dataset \\
	--level1-dimreducer $dr1 \\
	--level1-classifier $cls1 \\
	--level1-reduce-dim-to 4 \\
	--level2-dimreducer $dr2 \\
	--level2-classifier $cls2 \\
	--level2-reduce-dim-to 40 \\
	--cv-folds 10 \\
	--output output/zijian/t2/${dataset}.lv1.${dr1}.${cls1}.lv2.${dr2}.${cls2}.json

deactivate"
			done
		done
	done
done
