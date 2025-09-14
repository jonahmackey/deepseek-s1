# Improving GRPO Sample Efficiency with Budget Forcing

This repository contains the code for our project:

**Improving GRPO Data Efficiency with Budget Forcing**  
[Jonah Mackey](https://jonahmackey.github.io/), University of Toronto / Vector Institute  
Yuchong Zhang, University of Toronto / Vector Institute  

* [Project Report (PDF)](report.pdf)  
* [Blog post](https://jonahmackey.github.io/blog/2025/grpo/)  

## Overview

Reinforcement learning (RL) fine-tuning has shown strong effectiveness for inducing reasoning in large language models, but it is very data-intensive. In this project, we explore **budget forcing** — enforcing a minimum token budget model generation — as a way to improve the data efficiency of RL fine-tuning for reasoning. 

Our method applies budget forcing during the sampling step of Group Relative Policy Optimization (GRPO) training. The intuition is that by forcing models to “think longer” before producing an answer, they may generate more correct responses per problem, yielding richer training signals and potentially accelerating learning.

In practice, we found that budget forcing increased reward signal in the early stages of training, but the advantage diminished over time and ultimately underperformed compared to the baseline.

## Repository Structure

The main set of files are in [grpo](./grpo):
* `grpo/train.py`: Runs GRPO training with optional budget forcing, including model setup (Unsloth + vLLM), dataset preparation, reward functions, training loop, checkpoint saving, and evaluation.
* `grpo/eval.py`: Evaluates trained checkpoints on GSM8K or MATH, running batched generation, extracting answers, computing accuracy/scores, and saving detailed logs/results.
* `grpo/reward.py`: Defines rule-based reward functions for RL training, including correctness checks and format-based rewards.
* `grpo/data.py`: Provides dataset loaders and preprocessing, including prompt formatting, answer extraction, and dataset splitting.
* `grpo/budget_forcing.py`: Implements the `WaitLogitsProcessor`, a custom logits processor that detects "</think>" tokens and forces the model to continue thinking by appending "Wait" until a minimum token budget is reached.

## Acknowledgements 

This project was built on top of the [Vector Institute GRPO reference implementation](https://github.com/VectorInstitute/vector-trl-references)
, created by Jacob Junqi Tian and John Willes. Their work integrated Unsloth with vLLM to enable GRPO with LoRA fine-tuning, making it practical to run GRPO experiments on the Vector cluster.

**Note on dependencies:** We do not provide a requirements.txt file because this code was developed and run on the [Vector Institute](https://vectorinstitute.ai/) cluster, where a pre-built environment (Ubuntu 22.04 Singularity image with unsloth, vllm, torch[cuda], and related packages) was already available. If you want to reproduce these experiments elsewhere, you’ll need to manually install these dependencies.