#!/bin/bash

log_dir=".log" # ouput log directory
out_dir="output/oxford/t1"
mkdir -p $log_dir
mkdir -p $out_dir

for dataset in {oxford-exponential,oxford-platform-1,oxford-platform-2}; do
	for dr in {none,kpca,lda,ism_sdr,pca,sup_pca}; do
		#for cls in {gnb,knn,lda,lr,svm_lin,svm_rbf}; do
		#	alloc_param="-p short -N1 -c1 --mem 4G --time 4:00:00"
		for cls in {svm_lin_cv,svm_rbf_cv}; do
			alloc_param="-p short -N1 -c8 --mem 8G --time 24:00:00"
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
	--reduce-dim-to 24 \\
	--cv-folds 10 \\
	--output ${out_dir}/${dataset}.${dr}.${cls}.${round}.json

deactivate"
			done
		done
	done
done
