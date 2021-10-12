#!/bin/bash

mkdir -p .log # ouput log directory
mkdir -p output/oxford/t1

for dataset in {oxford-exponential,oxford-platform-1,oxford-platform-2}; do
	for dr in {none,kpca,lda,ism_sdr,pca,sup_pca}; do
		for cls in {gnb,lda,lr,svm_lin,svm_rbf,svm_lin_cv,svm_rbf_cv}; do
			for round in $(seq 0 9); do
				job_desc="$dataset.$dr.$cls.$round"
				sbatch -J $job_desc \
					-o ".log/"$job_desc".log" \
					-e ".log/"$job_desc".err" \
					-p short -N1 -c4 --mem 4G --time 24:00:00 \
					--wrap \
"# run experiments #
echo \$SLURM_JOB_ID >&2
module load python/3.7.1
. /home/li.gua/python-virtual-env/python-3.7.1-metagenomic-general/bin/activate

python3 ./script/t1_methods.py \\
	--dataset $dataset \\
	--classifier $cls \\
	--dimreducer $dr \\
	--reduce-dim-to 26 \\
	--cv-folds 10 \\
	--output output/oxford/t1/${dataset}.${dr}.${cls}.${round}.json

deactivate"
			done
		done
	done
done
