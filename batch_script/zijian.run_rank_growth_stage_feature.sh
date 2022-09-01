#!/bin/bash

log_dir=".log" # ouput log directory
out_dir="output/zijian/rank_growth_stage_feature"
mkdir -p $log_dir
mkdir -p $out_dir

dataset="zijian-phase-and-strain"
for s in $(cut -f2 data/zijian_40.labels.txt | uniq | sort | uniq); do
	#for m in {fisher_score,hsic,lap_score,trace_ratio}; do
	for m in {lap_score,}; do
		n_cores="8"
		alloc_param="-p short -N1 -c$n_cores --mem 16G --time 24:00:00"
		job_desc="$dataset.$m.$s"
		sbatch -J $job_desc \
			-o "$log_dir/"$job_desc".log" \
			-e "$log_dir/"$job_desc".err" \
			$alloc_param "$@" \
			--wrap \
"# run experiments #
echo \$SLURM_JOB_ID >&2
. /home/li.gua/.local/env/python-3.10-venv/bin/activate

python3 ./script/rank_growth_stage_features_by_strain.py \\
	--dataset $dataset \\
	--rank-method $m \\
	--strain-list $s \\
	--output ${out_dir}/${job_desc}.tsv

deactivate"
	done
done
