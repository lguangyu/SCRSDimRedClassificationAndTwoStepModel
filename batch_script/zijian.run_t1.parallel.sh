#!/bin/bash

log_dir=".log" # ouput log directory
out_dir="output/zijian/t1"
mkdir -p $log_dir
mkdir -p $out_dir

for dataset in {zijian-exponential,zijian-stationary-1,zijian-stationary-2,zijian-stationary-3}; do
	for dr in {none,kpca,lda,ism_sdr,pca,sup_pca}; do
		for cls in {gnb,knn,lda,lr,rf,svm_lin,svm_rbf,svm_lin_cv,svm_rbf_cv,nn}; do
			n_cores="10"
			alloc_param="-p short -N1 -c$n_cores --mem 16G --time 24:00:00"
			for round in $(seq 0 9); do
				job_desc="$dataset.$dr.$cls.$round"
				sbatch -J $job_desc \
					-o "$log_dir/"$job_desc".log" \
					-e "$log_dir/"$job_desc".err" \
					$alloc_param "$@" \
					--wrap \
"# run experiments #
echo \$SLURM_JOB_ID >&2
. /home/li.gua/.local/env/python-3.10-venv/bin/activate

python3 ./script/t1_methods.py \\
	--dataset $dataset \\
	--dimreducer $dr \\
	--reduce-dim-to 35 \\
	--classifier $cls \\
	--cv-folds 10 \\
	--cv-parallel $n_cores \\
	--output ${out_dir}/${job_desc}.json

deactivate"
			done
		done
	done
done
