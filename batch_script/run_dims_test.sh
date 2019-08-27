#!/bin/bash

mkdir -p .log # ouput log directory

#for dataset in {exponential,platform-1,platform-2}; do
#for dataset in {platform-1,platform-2}; do
dataset="platform-2"
	for dr in {none,lda,lsdr,pca}; do
		for cls in {gnb,lda,lr,svm_lin,svm_rbf,svm_lin_cv,svm_rbf_cv}; do
			for nd in $(seq 2 40); do 
				job_desc="$dataset.$dr.$nd.$cls"; echo $job_desc
				sbatch -J $job_desc \
					-o ".log/"$job_desc".log" \
					-e ".log/"$job_desc".err" \
					-p general -N1 -c1 --mem 2G \
					--wrap \
"# run experiments #
echo \$SLURM_JOB_ID >&2
module load python/3.7.1
. /home/li.gua/python-virtual-env/python-3.7.1-metagenomic-general/bin/activate

mkdir -p output/dims_test
python3 ./script/t1_methods.py \\
	--dataset $dataset \\
	--classifier $cls \\
	--dimreducer $dr \\
	--reduce-dim-to $nd \\
	--cv-folds 10 \\
	--output output/dims_test/$job_desc.json

deactivate"
			done
		done
	done
#done
