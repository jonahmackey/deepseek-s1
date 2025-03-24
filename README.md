# Vector TRL Examples

This repository contains various examples of how to fine-tune LLMs using RL on the Vector Cluster.

## Quickstart

To quickly get started with GRPO fine-tuning on the Vector Cluster:

1. Clone the repository:
   ```bash
   git clone https://github.com/vectorInstitute/vector-trl-references.git
   cd vector-trl-references/
   ```

2. Submit the GRPO job:
   ```bash
   sbatch examples/grpo/unsloth_vllm_lora_grpo.slurm.sh
   ```

3. Monitor the logs:
   ```bash
   export SLURM_JOB_ID=<your_job_id>
   tail -f logs/grpo_${SLURM_JOB_ID}.out logs/grpo_${SLURM_JOB_ID}.err
   ```

That's it! The job will use our pre-built Singularity image with all dependencies included.

## Implementation details

This project combines several key technologies to enable efficient RL fine-tuning:

- **vLLM**: A high-throughput, memory-efficient inference engine for LLMs that provides efficient rollout capabilities. It uses PagedAttention for optimized memory management.

- **Unsloth with LoRA**: Unsloth lowers the compute requriements by patching vLLM and TRL to work together efficiently. It also implements parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation).

- **TRL (Transformer Reinforcement Learning)**: A library that implements various online RL algorithms for LLMs, including GRPO (Generative Reinforcement Policy Optimization), PPO, and others.

- **Singularity**: A containerization solution for HPC environments that allows us to package all dependencies in a single image file (SIF).

## Minimum Resources

By default, the quickstart example runs on the following resources:
- 1 NVIDIA A40 GPU
- 8 CPU cores
- 32GB RAM
- 5 hours of runtime

You can modify these resource allocations by editing the SLURM parameters in `examples/grpo/unsloth_vllm_lora_grpo.slurm.sh` if needed. This implementation is intended to serve as a starting point for experimentation - full scale training is better suited for the A100 partition.


## FAQs

**Why is there's no package to install, and no 10GB+ Torch virtual environment to create?**

In order to save your disk quota, and avoid creating another 10GB+ virtual environment just to experiment with GRPO. We have bundled a number of dependencies (torch, vllm, unsloth, trl, transformer, datasets, wandb, etc.) into one pre-built Singularity Image (SIF) stored at a shared location on the cluster, so you can get started without taking up any space on your home directory.

See docs/source/building_sif.md to learn about how we built this image. However, if you are just trying to install another package, there's no need to build a new image! Read on:

**That's nice, but what if I need to install additional packages on top of the provided environment?**

Short answer: just add them to pyproject.toml under "dependencies".

Long answer: We have bundled the SIF image with uv. On the first run, the SLURM script that we provide will create a venv just for you under `.venv`. This venv is almost empty (~27KB) by default, and is overlaid on top of the 10GB of packages provided through the SIF image (vllm, etc.) When you add your custom dependencies to pyproject.toml under "dependencies", these dependencies are installed at the first run, and will be become available alongside existing packages provided through the SIF image.

**What packages are available in the SIF file?**

Quite a few. Run the following to see the full list of packages that are available.

```bash
# Run the command on a compute node, not on the login node.
srun -p cpu -c 1 --mem 2GB -t 1:00:00 --pty bash

export SIF_PATH=/projects/llm/unsloth-vllm-trl-latest.sif

# Create overlay venv, if not yet created.
module load singularity-ce
singularity exec ${SIF_PATH} uv venv --system-site-packages .venv
singularity exec ${SIF_PATH} uv run pip freeze
```

You should expect to see "vllm @ file:///vllm-workspace/dist/vllm-..." in the pip output.
