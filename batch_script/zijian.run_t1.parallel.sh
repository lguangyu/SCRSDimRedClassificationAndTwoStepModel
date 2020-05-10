#!/bin/bash

mkdir -p .log # ouput log directory

for dataset in {zijian-exponential,zijian-stationary-1,zijian-stationary-2,zijian-stationary-3}; do
	for dr in {none,kpca,lda,lsdr,pca,sup_pca}; do
		for cls in {gnb,lda,lr,svm_lin,svm_rbf,svm_lin_cv,svm_rbf_cv}; do
			job_desc="$dataset.$dr.$cls"
			sbatch -J $job_desc \
				-o ".log/"$job_desc".log" \
				-e ".log/"$job_desc".err" \
				-p short -N1 -c2 \
				--mem 4G --time 12:00:00 \
				--wrap \
"# run experiments #
echo \$SLURM_JOB_ID >&2
module load python/3.7.1
. /home/li.gua/python-virtual-env/python-3.7.1-metagenomic-general/bin/activate

python3 ./script/t1_methods.py \\
	--dataset $dataset \\
	--classifier $cls \\
	--dimreducer $dr \\
	--reduce-dim-to 40 \\
	--cv-folds 10 \\
	--output output/${dataset}.${dr}.${cls}.json

deactivate"
		done
	done
done
