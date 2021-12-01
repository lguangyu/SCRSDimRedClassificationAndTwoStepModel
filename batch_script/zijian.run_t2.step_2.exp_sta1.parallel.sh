#!/bin/bash

log_dir=".log" # ouput log directory
out_dir="output/zijian/t2/step_2-exp_sta1"
mkdir -p $log_dir
mkdir -p $out_dir

dataset="zijian-exp-sta1-phase-and-strain"
dr1=none
cls1=svm_rbf_man
cls1_disp=svm_rbf_cv
for dr2 in {none,kpca,lda,ism_sdr,pca,sup_pca}; do
	for cls2 in {gnb,knn,lda,lr,rf,svm_lin,svm_rbf,svm_lin_cv,svm_rbf_cv}; do
		n_cores="10"
		alloc_param="-p short -N1 -c$n_cores --mem 48G --time 24:00:00"
		for round in $(seq 0 9); do
			job_desc="$dataset.$dr1.$cls1_disp.$dr2.$cls2.$round"
			sbatch -J $job_desc \
				-o "$log_dir/"$job_desc".log" \
				-e "$log_dir/"$job_desc".err" \
				$alloc_param "$@" \
				--wrap \
"# run experiments #
echo \$SLURM_JOB_ID >&2
. /home/li.gua/.local/env/python-3.10-venv/bin/activate

python3 ./script/t2_methods.py \\
	--dataset $dataset \\
	--level1-dimreducer $dr1 \\
	--level1-reduce-dim-to 5 \\
	--level1-classifier $cls1 \\
	--level2-dimreducer $dr2 \\
	--level2-reduce-dim-to 35 \\
	--level2-classifier $cls2 \\
	--cv-folds 10 \\
	--cv-parallel $n_cores \\
	--output ${out_dir}/${job_desc}.json

deactivate"
		done
	done
done
