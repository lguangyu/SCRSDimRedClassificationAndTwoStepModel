#!/bin/bash

log_dir=".log" # ouput log directory
out_dir="output/oxford/t2/step_1"
mkdir -p $log_dir
mkdir -p $out_dir

dataset="oxford-phase-only"
for dr in {none,kpca,lda,ism_sdr,pca,sup_pca}; do
	for cls in {gnb,knn,lda,lr,rf,svm_lin,svm_rbf,svm_lin_cv,svm_rbf_cv}; do
		n_cores="10"
		alloc_param="-p short -N1 -c$n_cores --mem 8G --time 24:00:00"
		for round in $(seq 0 9); do
			job_desc="$dataset.$dr.$cls.$round"
			sbatch -J $job_desc \
				-o "$log_dir/"$job_desc".log" \
				-e "$log_dir/"$job_desc".err" \
				$alloc_param \
				--wrap \
"# run experiments #
echo \$SLURM_JOB_ID >&2
module load python/3.7.1
. /home/li.gua/python-virtual-env/python-3.7.1-metagenomic-general/bin/activate

python3 ./script/t1_methods.py \\
	--dataset $dataset \\
	--classifier $cls \\
	--dimreducer $dr \\
	--reduce-dim-to 5 \\
	--cv-folds 10 \\
	--cv-parallel $n_cores \\
	--output ${out_dir}/${dataset}.${dr}.${cls}.${round}.json

deactivate"
		done
	done
done
