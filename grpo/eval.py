from unsloth import FastLanguageModel, PatchFastRL
from grpo.data import get_gsm8k_questions, extract_xml_answer
from vllm import SamplingParams


PatchFastRL("GRPO", FastLanguageModel)


def evaluate_checkpoint(
    checkpoint_path: str,
    test_dataset,
    base_model_name: str = "/model-weights/Qwen2.5-3B-Instruct",
    max_seq_length: int = 4096,
    lora_rank: int = 64,
    sampling_params: SamplingParams = None,
):
    """
    Loads a base model and applies a LoRA checkpoint from the provided path,
    then evaluates the model on the gsm8k test dataset.
    
    Args:
        checkpoint_path (str): Path to the saved LoRA checkpoint.
        num_examples (int, optional): Number of test examples to evaluate. If None, uses all.
        base_model_name (str): Path or name of the base model.
        max_seq_length (int): Maximum sequence length for the model.
        lora_rank (int): Rank for the LoRA modules.
        sampling_params (SamplingParams): parameters for sampling.
    
    Returns:
        A tuple (avg_score, accuracy) where:
          - avg_score: The average score based on a simple correctness check.
          - accuracy: The percentage of examples for which the generated answer exactly matches the expected answer.
    """
    # Load the base model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,       # set to False if you prefer full precision (16-bit)
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.5,
    )
    
    model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
    )
    
    # Load the LoRA checkpoint from the given path
    lora_weights = model.load_lora(checkpoint_path)
    
    total_score = 0.0
    num_correct = 0
    
    # Loop through each test example
    for idx, example in enumerate(test_dataset):
        # Prepare the prompt (assumes prompt is a list of messages)
        input_text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate output using fast_generate; note that we pass the lora weights
        output = (
            model.fast_generate(
                input_text,
                sampling_params=sampling_params,
                lora_request=lora_weights,
            )[0]
            .outputs[0]
            .text
        )
        
        # Extract the answer using the helper function (assumes <think> and <answer> XML format)
        generated_answer = extract_xml_answer(output)
        expected_answer = example["answer"]
        
        # Evaluate the output: using a simple exact match (like your correctness_reward_func logic)
        is_correct = (generated_answer.strip() == expected_answer.strip())
        score = 2.0 if is_correct else 0.0
        
        total_score += score
        if is_correct:
            num_correct += 1
        
        # Print per-example details
        if idx < 50:
            print(f"Example {idx + 1}:")
            print(f"Question: {example['prompt'][-1]['content']}")
            print(f"Expected Answer: {expected_answer}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Score: {score}")
            print("-" * 50)
    
    # Compute average score and accuracy
    avg_score = total_score / len(test_dataset) if test_dataset else 0.0
    accuracy = num_correct / len(test_dataset) if test_dataset else 0.0
    print(f"Average Score: {avg_score}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return avg_score, accuracy


def evaluate_built_model(
        model,
        tokenizer,
        loaded_lora,
        test_dataset,
        sampling_params: SamplingParams = None,
        
    ):
    """
    This function is the same as evaluate_checkpoint except it takes in an already
    built model and tokenizer.
    
    - model and tokenizer should be obtained by:
        "model, tokenizer = FastLanguageModel.from_pretrained(....)"
        "model = FastLanguageModel.get_peft_model(model, ...)"
    
    - loaded_lora should be obtained by something like 
        "model.load_lora(path/to/checkpoint)"
    
    """
    total_score = 0.0
    num_correct = 0
    
    # Loop through each test example
    for idx, example in enumerate(test_dataset):
        # Prepare the prompt (assumes prompt is a list of messages)
        input_text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate output using fast_generate; note that we pass the lora weights
        output = (
            model.fast_generate(
                input_text,
                sampling_params=sampling_params,
                lora_request=loaded_lora,
            )[0]
            .outputs[0]
            .text
        )
        
        # Extract the answer using the helper function (assumes <think> and <answer> XML format)
        generated_answer = extract_xml_answer(output)
        expected_answer = example["answer"]
        
        # Evaluate the output: using a simple exact match (like your correctness_reward_func logic)
        is_correct = (generated_answer.strip() == expected_answer.strip())
        score = 2.0 if is_correct else 0.0
        
        total_score += score
        if is_correct:
            num_correct += 1
        
        # Print per-example details
        if idx < 50:
            print(f"Example {idx + 1}:")
            print(f"Question: {example['prompt'][-1]['content']}")
            print(f"Expected Answer: {expected_answer}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Score: {score}")
            print("-" * 50)
    
    # Compute average score and accuracy
    avg_score = total_score / len(test_dataset) if test_dataset else 0.0
    accuracy = num_correct / len(test_dataset) if test_dataset else 0.0
    print(f"Average Score: {avg_score}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return avg_score, accuracy


# Example usage:


# For evaluate_checkpoint:
# test_dataset = get_gsm8k_questions(split="test", num_examples=1024)
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
# avg_score, acc = evaluate_checkpoint("/h/yuchongz/vector-trl-references/grpo_saved_lora", test_dataset=test_dataset, sampling_params=sampling_params)


# For evaluate_built_model
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="/model-weights/Qwen2.5-3B-Instruct",
#     max_seq_length=4096,
#     load_in_4bit=True,       # set to False if you prefer full precision (16-bit)
#     fast_inference=True,
#     max_lora_rank=64,
#     gpu_memory_utilization=0.5,
# )
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ],  # Remove QKVO if out of memory
#     lora_alpha=64,
#     use_gradient_checkpointing="unsloth",  # Enable long context finetuning
#     random_state=3407,
#     )

# lora_weights = model.load_lora("/h/yuchongz/vector-trl-references/grpo_saved_lora")
# evaluate_built_model(model, tokenizer, lora_weights, test_dataset, sampling_params)
