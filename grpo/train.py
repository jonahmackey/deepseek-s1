import sys

sys.path = [
    "",
    ".venv/lib/python3.12/site-packages",
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
    "/usr/local/lib/python3.12/dist-packages",
    "/usr/lib/python3/dist-packages",
]

from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from grpo.data import get_dataset
from grpo.budget_forcing import WaitLogitsProcessor
from grpo.reward import get_reward_funcs, get_format_reward_funcs
from grpo.eval import evaluate_built_model
from pathlib import Path

PatchFastRL("GRPO", FastLanguageModel)


def vLLMSamplingParams(**kwargs):
    attach_kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = attach_kwargs
    return sampling_params


def run(args):
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    # Load up `Qwen 2.5 3B Instruct`, and set parameters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_completion_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=0.5,  # Reduce if out of memory
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )
    
    # Data prep
    # train_dataset = get_gsm8k_questions(split="train", num_examples=args.num_examples)
    train_dataset = get_dataset(task=args.task, split="train")
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Setup budget forcing  
    if args.min_budget > 0:
        # TODO generalized next token id
        logits_processors = [WaitLogitsProcessor(tokenizer, device=DEVICE, next_token_id=14190, min_num_tokens=args.min_budget)]
    else:
        logits_processors = None
        
    vllm_sampling_params = vLLMSamplingParams(max_tokens=args.max_completion_length,
                                              logits_processors=logits_processors)
        
    # GRPO Trainer
    model_name = args.model_name.split("/")[-1]
    run_name = f"{model_name}-{args.task}-n={args.num_generations}-b={args.per_device_train_batch_size}-g={args.gradient_accumulation_steps}-max={args.max_completion_length}-bf={args.min_budget}"
    if args.format_reward:
        run_name += "-format"
    run_name += "-final"
    
    training_args = GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference!
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Increase to 4 for smoother training
        num_generations=args.num_generations,  # Decrease if out of memory
        max_prompt_length=256,
        max_completion_length=args.max_completion_length,
        num_train_epochs=1,  # Set to 1 for a full training run
        max_steps=args.num_steps,
        save_steps=args.num_steps,
        max_grad_norm=0.1,
        report_to="wandb",  # Can use Weights & Biases
        output_dir=f"./outputs/{run_name}",
        run_name=run_name,
        vllm_sampling_params=vllm_sampling_params,
    )
    
    reward_funcs = get_reward_funcs(args.task)
    if args.format_reward:
        reward_funcs += get_format_reward_funcs()
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
        )
    trainer.train()
    
    checkpoint_save_path = Path(f"{args.save_path}/checkpoints/{run_name}")
    checkpoint_save_path.mkdir(parents=True, exist_ok=True)
    model.save_lora(checkpoint_save_path)
    loaded_lora = model.load_lora(f'outputs/{run_name}/checkpoint-{args.num_steps}')
    test_dataset = get_dataset(task=args.task, split="test")
    eval_sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=args.max_completion_length)
    
    evaluate_built_model(model, tokenizer, loaded_lora, test_dataset, eval_sampling_params, batch_size=128)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GRPO training.")
    parser.add_argument("--model_name", type=str, default="/model-weights/Qwen2.5-3B-Instruct")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=250)
    parser.add_argument("--format_reward", action="store_true")
    parser.add_argument("--min_budget", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="./")
    args = parser.parse_args()
    
    run(args)