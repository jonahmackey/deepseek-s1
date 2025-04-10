from unsloth import FastLanguageModel, PatchFastRL
from grpo.data import get_gsm8k_questions, extract_xml_answer
from vllm import SamplingParams

PatchFastRL("GRPO", FastLanguageModel)

def evaluate_checkpoint(
    checkpoint_path: str,
    test_dataset: dict,
    base_model_name: str = "/model-weights/Qwen2.5-3B-Instruct",
    max_seq_length: int = 4096,
    lora_rank: int = 64,
    sampling_params: SamplingParams = None,
    batch_size: int = 32,
):
    """
    Loads a base model and applies a LoRA checkpoint from the provided path,
    then evaluates the model on the gsm8k test dataset (which is a dict with keys
    "prompt", "answer", and "question") in batches.

    Args:
        checkpoint_path (str): Path to the saved LoRA checkpoint.
        test_dataset (dict): Dictionary with keys "prompt", "answer", and "question".
            - "prompt": list of prompts (each prompt is a list of messages).
            - "answer": list of expected answer strings.
            - "question": list of plain question strings for printing.
        base_model_name (str): Path or name of the base model.
        max_seq_length (int): Maximum sequence length.
        lora_rank (int): Rank for LoRA modules.
        sampling_params (SamplingParams): Parameters for sampling.
        batch_size (int): Batch size for evaluation (default is 32).

    Returns:
        A tuple (avg_score, accuracy) where:
            - avg_score: The average score (2.0 for a correct answer, 0.0 otherwise).
            - accuracy: The proportion of examples with an exact match.
    """
    # Load the base model and tokenizer.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,       # Set to False if you prefer full precision (16-bit)
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.5,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Load the LoRA checkpoint.
    lora_weights = model.load_lora(checkpoint_path)
    
    total_score = 0.0
    num_correct = 0
    total_examples = len(test_dataset["answer"])
    
    # Process the dataset in batches.
    for batch_start in range(0, total_examples, batch_size):
        # Slice the batch from each key.
        batch_prompts = test_dataset["prompt"][batch_start: batch_start + batch_size]
        batch_answers = test_dataset["answer"][batch_start: batch_start + batch_size]
        batch_questions = test_dataset["question"][batch_start:batch_start+batch_size] if "question" in test_dataset else test_dataset["problem"][batch_start:batch_start+batch_size]        
        # Prepare input texts for the batch.
        batch_inputs = []
        for prompt in batch_prompts:
            # prompt is assumed to be a list of messages (each a dict with a "content" key)
            input_text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_inputs.append(input_text)
        
        # Generate outputs for the entire batch.
        batch_outputs = model.fast_generate(
            batch_inputs,
            sampling_params=sampling_params,
            lora_request=lora_weights,
        )
        
        # Process each output.
        for idx, (expected_answer, question_text, generation_result) in enumerate(
            zip(batch_answers, batch_questions, batch_outputs)
        ):
            output_text = generation_result.outputs[0].text
            generated_answer = extract_xml_answer(output_text)
            
            is_correct = (generated_answer.strip() == expected_answer.strip())
            score = 2.0 if is_correct else 0.0
            total_score += score
            if is_correct:
                num_correct += 1
            
            overall_idx = batch_start + idx
            if overall_idx < 50:
                print(f"Example {overall_idx + 1}:")
                print(f"Question: {question_text}")
                print(f"Expected Answer: {expected_answer}")
                print(f"Generated Answer: {generated_answer}")
                print(f"Score: {score}")
                print("-" * 50)
    
    avg_score = total_score / total_examples if total_examples > 0 else 0.0
    accuracy = num_correct / total_examples if total_examples > 0 else 0.0
    print(f"Average Score: {avg_score}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return avg_score, accuracy


def evaluate_built_model(
    model,
    tokenizer,
    loaded_lora,
    test_dataset: dict,
    sampling_params: SamplingParams = None,
    batch_size: int = 32,
    compare_function: callable = None,
    task: str = "math",
):
    """
    Evaluates an already built model (with its tokenizer and loaded LoRA checkpoint)
    on the gsm8k test dataset (dict with keys "prompt", "answer", and "question") in batches.

    Args:
        model: The loaded model.
        tokenizer: The model's tokenizer.
        loaded_lora: Loaded LoRA checkpoint weights.
        test_dataset (dict): Dictionary with keys "prompt", "answer", and "question".
        sampling_params (SamplingParams): Parameters for sampling.
        batch_size (int): Batch size for evaluation (default is 32).

    Returns:
        A tuple (avg_score, accuracy) where:
            - avg_score: The average score (2.0 for a correct answer, 0.0 otherwise).
            - accuracy: The proportion of examples with an exact match.
    """
    total_score = 0.0
    num_correct = 0
    total_examples = len(test_dataset["answer"])
    
    for batch_start in range(0, total_examples, batch_size):
        batch_prompts = test_dataset["prompt"][batch_start: batch_start + batch_size]
        batch_answers = test_dataset["answer"][batch_start: batch_start + batch_size]
        batch_questions = test_dataset["question"][batch_start:batch_start+batch_size] if "question" in test_dataset else test_dataset["problem"][batch_start:batch_start+batch_size]
        
        batch_inputs = []
        for prompt in batch_prompts:
            input_text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_inputs.append(input_text)
        
        batch_outputs = model.fast_generate(
            batch_inputs,
            sampling_params=sampling_params,
            lora_request=loaded_lora,
        )
        
        for idx, (expected_answer, question_text, generation_result) in enumerate(
            zip(batch_answers, batch_questions, batch_outputs)
        ):
            output_text = generation_result.outputs[0].text
            
            if compare_function is None:
                generated_answer = extract_xml_answer(output_text)
                
                is_correct = (generated_answer.strip() == expected_answer.strip())
            else:
                is_correct = compare_function(output_text, expected_answer)
                assert isinstance(is_correct, bool)
            
            if is_correct:
                if task == "math":
                    score = 1.0
                else:
                    score = 2.0
            else:
                score = 0.0
            total_score += score
            if is_correct:
                num_correct += 1
            
            overall_idx = batch_start + idx
            if overall_idx < 50:
                print(f"Example {overall_idx + 1}:")
                print(f"Question: {question_text}")
                print(f"Expected Answer: {expected_answer}")
                print(f"Generated Answer: {output_text}")
                print(f"Score: {score}")
                print("-" * 50)
    
    avg_score = total_score / total_examples if total_examples > 0 else 0.0
    accuracy = num_correct / total_examples if total_examples > 0 else 0.0
    print(f"Average Score: {avg_score}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return avg_score, accuracy


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # Make sure you have the 'peft' package installed

def evaluate_checkpoint_hf(
    checkpoint_path: str,
    test_dataset: dict,
    base_model_name: str = "/model-weights/Qwen2.5-3B-Instruct",
    max_seq_length: int = 4096,
    batch_size: int = 32,
    sampling_kwargs: dict = None,
):
    """
    Loads a base model from Hugging Face Transformers, applies a LoRA adapter (using the PEFT library),
    and evaluates the model on the provided test dataset.

    Args:
        checkpoint_path (str): The path to the saved LoRA checkpoint directory 
            (should contain at least adapter_config.json and adapter_model.safetensors).
        test_dataset (dict): Dictionary with keys "prompt", "answer", and "question":
            - "prompt": List of prompts, where each prompt is a list of messages (each a dict with "role" and "content").
            - "answer": List of expected answer strings.
            - "question": List of plain question strings (for printing/logging).
        base_model_name (str): The Hugging Face model identifier or local path for the base model.
        max_seq_length (int): Maximum token length for input and generation.
        batch_size (int): Evaluation batch size.
        sampling_kwargs (dict): Optional dictionary with additional generation parameters (e.g., do_sample, temperature).

    Returns:
        tuple: (avg_score, accuracy) where:
            - avg_score: The average score (2.0 for a correct answer, 0.0 otherwise).
            - accuracy: The proportion of examples with an exact match.
    """
    import os
    print(os.listdir(checkpoint_path))
    # Default sampling parameters (customize as needed)
    if sampling_kwargs is None:
        sampling_kwargs = {
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
        }
    
    # Load the tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        # If you want to use mixed-precision, you can specify torch_dtype:
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    
    # Load the LoRA adapter weights via the PEFT library
    model = PeftModel.from_pretrained(model, checkpoint_path, torch_dtype=torch.float16)
    
    # Send model to device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    total_score = 0.0
    num_correct = 0
    total_examples = len(test_dataset["answer"])

    # Loop over the test dataset in batches
    for i in range(0, total_examples, batch_size):
        # Extract the current batch for each key
        batch_prompts = test_dataset["prompt"][i:i+batch_size]
        batch_answers = test_dataset["answer"][i:i+batch_size]
        batch_questions = test_dataset["question"][i:i+batch_size] if "question" in test_dataset else test_dataset["problem"][i:i+batch_size]
        
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
            max_length=max_seq_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate outputs using the model
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_seq_length,
                **sampling_kwargs
            )
        # Decode the generated texts
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Evaluate each example in the batch
        for j, (gen_text, expected_answer, question_text) in enumerate(zip(generated_texts, batch_answers, batch_questions)):
            # For simplicity, assume the generated answer is the text after the last newline.
            # You might need to adapt this extraction depending on your expected format.
            generated_answer = gen_text.strip().split("\n")[-1]
            
            is_correct = (generated_answer.strip() == expected_answer.strip())
            score = 2.0 if is_correct else 0.0
            total_score += score
            num_correct += 1 if is_correct else 0
            
            overall_idx = i + j
            if overall_idx < 50:  # Print out details for the first 50 examples
                print(f"Example {overall_idx + 1}:")
                print(f"Question: {question_text}")
                print(f"Expected Answer: {expected_answer}")
                print(f"Generated Answer: {generated_answer}")
                print(f"Score: {score}")
                print("-" * 50)
    
    avg_score = total_score / total_examples if total_examples > 0 else 0.0
    accuracy = num_correct / total_examples if total_examples > 0 else 0.0
    print(f"Average Score: {avg_score}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return avg_score, accuracy



# test_dataset = get_gsm8k_questions(split="test")
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
# avg_score, acc = evaluate_checkpoint("/h/yuchongz/deepseek-s1/checkpoints/n=8-b=8-g=1-max=4096", test_dataset=test_dataset, sampling_params=sampling_params, batch_size=128)
