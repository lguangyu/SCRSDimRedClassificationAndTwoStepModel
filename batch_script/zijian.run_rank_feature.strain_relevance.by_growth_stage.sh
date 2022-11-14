#!/bin/bash

log_dir=".log" # ouput log directory
out_dir="output/zijian/rank_feature"
mkdir -p $log_dir
mkdir -p $out_dir

dataset="zijian-phase-and-strain"
for m in {fisher_score,hsic,lap_score,trace_ratio}; do
	n_cores="8"
	alloc_param="-p short -N1 -c$n_cores --mem 16G --time 24:00:00"
	job_desc="$dataset.strain_relevance.by_growth_stage.$m"
	sbatch -J $job_desc \
		-o "$log_dir/"$job_desc".log" \
		-e "$log_dir/"$job_desc".err" \
		$alloc_param "$@" \
		--wrap \
"# run experiments #
echo \$SLURM_JOB_ID >&2
. /home/li.gua/.local/env/python-3.10-venv/bin/activate

python3 ./script/rank_feature.growth_stage_relevance.by_strain.py \\
	--dataset $dataset \\
	--rank-method $m \\
	--output ${out_dir}/${job_desc}.tsv

deactivate"
done
