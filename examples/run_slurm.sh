#!/usr/bin/env bash
#SBATCH -J tehood
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -o log/tehood.%A_%a.out
# One genome per array task:
#   mkdir -p log
#   sbatch --array=1-$(($(wc -l < samples.tsv) - 1))%20 examples/run_slurm.sh samples.tsv results
set -euo pipefail
tehood run --samples "${1:?samples.tsv}" --outdir "${2:-results}" --row "$SLURM_ARRAY_TASK_ID"
