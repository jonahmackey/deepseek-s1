from eval import evaluate_checkpoint
import argparse
from data import get_gsm8k_questions
from vllm import SamplingParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval GRPO LoRA model.")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--base_model_name", type=str, default="/model-weights/Qwen2.5-3B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=64)
    
    args = parser.parse_args()
    
    test_dataset = get_gsm8k_questions(split="test")
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
    
    evaluate_checkpoint(args.checkpoint_path, test_dataset, args.base_model_name, args.max_seq_length, args.lora_rank, sampling_params)
    
    