# Vector TRL Examples (uv, vLLM, Unsloth, singularity)

```bash
salloc -p a40 --gres gpu:1 -c 8 --mem 32GB --time 13:00:00 --job-name vllm-trl

module load singularity-ce

singularity exec \
--nv \
--bind /model-weights:/model-weights \
--bind /projects/llm:/projects/llm \
--bind /scratch/ssd004/scratch/`whoami`:/scratch \
--bind /opt/slurm/:/opt/slurm/ \
/projects/llm/unsloth-vllm-trl-latest.sif \
python3 /h/jacobtian/r1/trl/examples/scripts/unsloth_grpo.py
```
