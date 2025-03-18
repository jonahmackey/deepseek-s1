#!/bin/bash
#SBATCH -p a40
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32GB
#SBATCH --time 5:00:00
#SBATCH --job-name vllm-trl
#SBATCH --output=logs/grpo_%A.out
#SBATCH --error=logs/grpo_%A.err

module load singularity-ce

SIF_PATH=/projects/llm/unsloth-vllm-trl-latest.sif

# Initialize overlay virtual environment on top of system packages from SIF
# This .venv is empty (~27KB) by default.
# To install new packages, modify pyproject.toml. See README.md FAQ for more details.
singularity exec \
${SIF_PATH} \
uv venv --system-site-packages .venv

singularity exec \
--nv \
--bind /model-weights:/model-weights \
--bind /projects/llm:/projects/llm \
--bind /scratch/ssd004/scratch/`whoami`:/scratch \
--bind /opt/slurm/:/opt/slurm/ \
${SIF_PATH} \
uv run examples/grpo/unsloth_vllm_lora_grpo.py
