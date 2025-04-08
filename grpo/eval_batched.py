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
        batch_questions = test_dataset["question"][batch_start: batch_start + batch_size]
        
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
        batch_questions = test_dataset["question"][batch_start: batch_start + batch_size]
        
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



# test_dataset = get_gsm8k_questions(split="test")
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
# avg_score, acc = evaluate_checkpoint("/h/yuchongz/deepseek-s1/checkpoints/n=8-b=8-g=1-max=4096", test_dataset=test_dataset, sampling_params=sampling_params, batch_size=128)
