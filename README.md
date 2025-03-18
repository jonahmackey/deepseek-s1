# Vector TRL Examples

This repository contains various examples of how to fine-tune LLMs using RL on the Vector Cluster.

## GRPO (uv, vLLM, Unsloth-TRL, singularity)

Why this combination?

- vLLM: Efficient Rollout.
- Unsloth: patches vLLM and TRL, so that vLLM would reuse the model weights from TRL- no need a separate copy in memory.
- TRL: implements various online RL algorithms, including GRPO.

How to run this example?

```bash
# Vaughan cluster
git clone https://github.com/vectorInstitute/vector-trl-references.git
cd vector-trl-references/
sbatch examples/grpo/unsloth_vllm_lora_grpo.slurm.sh
```

Note that depending on the model size, it might take some time to see the "a-ha" moment of self-correction, and you might need to modify the hyperparameters listed in unsloth_vllm_lora_grpo.py where appropriate. We are trying to figure out a way to make the behavior more consistent, and we welcome your input on how to make that happen.

## FAQs

> Why is there's no package to install, and no 10GB+ Torch virtual environment to create?

We very much understand that you might be conscious about your disk quota, and would prefer not to create another 10GB+ virtual environment just to experiment with GRPO. Thus, we have bundled a number of dependencies (torch, vllm, unsloth, trl, transformer, datasets, wandb, etc.) into one pre-built Singularity Image (SIF) stored at a shared location on the cluster, so you can get started without taking up any space on your home directory.

See docs/source/building_sif.md to learn about how we built this image. However, if you are just trying to install another package, there's no need to build a new image! Read on:

> That's nice, but what if I need to install additional packages on top of the provided environment?

Short answer: just add them to pyproject.toml under "dependencies".

Long answer: We have bundled the SIF image with astral-uv. On the first run, the SLURM script that we provide will create a venv just for you under `.venv`. This venv is almost empty (~27KB) by default, and is overlaid on top of the 10GB of packages provided through the SIF image (vllm, etc.) When you add your custom dependencies to pyproject.toml under "dependencies", these dependencies are installed at the first run, and will be become available alongside existing packages provided through the SIF image.

> What packages are available in the SIF file?

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
