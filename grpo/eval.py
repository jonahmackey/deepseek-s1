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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # Make sure you have the 'peft' package installed
from grpo.data import get_dataset, extract_xml_answer
from pathlib import Path

def eval_checkpoint(
    checkpoint_path: str,
    task: str,
    base_model_name: str,
    batch_size: int = 32,
    num_examples: int = -1,
):
    # Sampling params for generation
    sampling_kwargs = {
            "do_sample": True,
            "top_p": 1.0,
            "temperature": 0.9,
        }
    
    # Load the tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    
    # Load the LoRA adapter weights via the PEFT library
    model = PeftModel.from_pretrained(model, checkpoint_path, torch_dtype=torch.float16)
    
    # Move model to GPU 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # get dataset
    test_dataset = get_dataset(task=task, split="test", num_examples=num_examples)
    
    total_score = 0.0
    num_correct = 0
    total_examples = len(test_dataset["answer"])
    
    # Logging
    run_name = checkpoint_path.split("/")[-2]
    # print(f"Evaluation Results - {run_name}\n")
    # print("=" * 50 + "\n\n")
    
    eval_save_path = Path(f"./eval_results/{run_name}")
    eval_save_path.mkdir(parents=True, exist_ok=True)
    output_filename = f"./eval_results/{run_name}/evaluation_results.txt"
    
    with open(output_filename, "w") as logfile:
        logfile.write("Evaluation Results\n")
        logfile.write("=" * 50 + "\n\n")
    
    # Loop over the test dataset in batches
    for i in range(0, total_examples, batch_size):
        # Extract the current batch for each key
        batch_prompts = test_dataset["prompt"][i:i+batch_size]
        batch_answers = test_dataset["answer"][i:i+batch_size]
        batch_questions = test_dataset["question"][i:i+batch_size]
        
        # Convert each prompt (list of messages) to a single input text string.
        # This example simply concatenates messages with a newline separator.
        input_texts = []
        for messages in batch_prompts:
            # You could enhance this further to include more context if needed.
            text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
            input_texts.append(text)
        
        # Tokenize the inputs
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate outputs using the model
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=4096,
                **sampling_kwargs
            )
        print("DONE")    
            
        # Decode the generated texts
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Evaluate each example in the batch
        for j, (gen_text, expected_answer, question_text) in enumerate(zip(generated_texts, batch_answers, batch_questions)):
            # For simplicity, assume the generated answer is the text after the last newline.
            # You might need to adapt this extraction depending on your expected format.
            generated_answer = gen_text
            
            generated_answer = extract_xml_answer(generated_answer)
            is_correct = generated_answer == expected_answer
            
            if is_correct:
                if task == "math":
                    score = 1.0
                elif task == "gsm8k":
                    score = 2.0
            else:
                score = 0.0
            total_score += score
            if is_correct:
                num_correct += 1
            
            overall_idx = i + j
                        
            # Logging
            if overall_idx < 30:
                print(f"Example {overall_idx + 1}:\n")
                print(f"Question: {question_text}\n")
                print("Prompt:\n")
                print(input_texts[j] + "\n")
                print(f"Expected Answer: {expected_answer}\n")
                print(f"Full Generated Response: {gen_text}\n")
                print(f"Extracted Answer: {generated_answer}\n")
                print(f"Score: {score}\n")
                print("-" * 50 + "\n")            

            # Save each example's details to the output file
            with open(output_filename, "a") as logfile:
                logfile.write(f"Example {overall_idx + 1}:\n")
                logfile.write(f"Question: {question_text}\n")
                logfile.write("Prompt:\n")
                logfile.write(input_texts[j] + "\n")
                logfile.write(f"Expected Answer: {expected_answer}\n")
                logfile.write(f"Full Generated Response: {gen_text}\n")
                logfile.write(f"Extracted Answer: {generated_answer}\n")
                logfile.write(f"Score: {score}\n")
                logfile.write("-" * 50 + "\n")
    
    avg_score = total_score / total_examples if total_examples > 0 else 0.0
    accuracy = num_correct / total_examples if total_examples > 0 else 0.0
    
    # Logging
    print("\n" + "=" * 50 + "\n")
    print(f"Average Score: {avg_score}\n")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    # Write summary metrics to the output file
    with open(output_filename, "a") as logfile:
        logfile.write("\n" + "=" * 50 + "\n")
        logfile.write(f"Average Score: {avg_score}\n")
        logfile.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    return avg_score, accuracy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run eval.")
    parser.add_argument("--checkpoint_path", type=str, default="/scratch/ssd004/scratch/jonah/large-models/outputs/Qwen2.5-3B-Instruct-gsm8k-n=16-b=16-g=4-max=4096-bf=256-final/checkpoint-300")
    parser.add_argument("--base_model_name", type=str, default="/model-weights/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_examples", type=int, default=512)
    args = parser.parse_args()
    
    eval_checkpoint(
        checkpoint_path=args.checkpoint_path,
        task=args.task,
        base_model_name=args.base_model_name,
        batch_size=args.batch_size,
        num_examples=args.num_examples,
    )