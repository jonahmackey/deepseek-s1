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

import re

from grpo.data import extract_xml_answer


def get_reward_funcs(task) -> list[callable]:
    if task == "gsm8k":
        return [correctness_reward_func]
    elif task == "countdown":
        return [countdown_correctness_reward_func]
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks are 'gsm8k' and 'countdown'.")

def get_format_reward_funcs() -> list[callable]:
    return [
        bf_strict_format_reward_func,
        bf_soft_format_reward_func,
        bf_xmlcount_reward_func,
    ]

# GSM8K
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# Format rewards for budget forcing
def bf_strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n([\s\S]*)</think>\n<answer>\n(.*?)\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def bf_soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>([\s\S]*)(?=</think>\s*<answer>)</think>\s*<answer>(.*?)</answer>"
    pattern = r"<think>([\s\S]*)(?=</think>\s*<answer>)</think>\s*<answer>(.*?)</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def bf_count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") > 0:
        count += 0.125
    if text.count("\n</think>\n") > 0:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def bf_xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [bf_count_xml(c) for c in contents]


# Countdown 
def extract_solution(response):
    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>([\s\S]*?)</answer>'
    match = re.finditer(answer_pattern, response)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False
   
    
def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


# NOTE created by jbxnes
def countdown_extraction_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    equations = [extract_solution(r) for r in responses]
    rewards = [0.0 if eqn is None else 0.5 for eqn in equations]
    return rewards


def countdown_valid_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    numbers = [ground_truth[i]['numbers'] for i in range(len(ground_truth))]
    
    equations = [extract_solution(r) for r in responses]
    rewards = [0.5 if validate_equation(eqn, nums) else 0.0 for eqn, nums in zip(equations, numbers)]
    return rewards
   
    
def countdown_correctness_reward_func(prompts, completions, ground_truth, **kwargs) -> list[float]:
    # responses
    responses = [completion[0]["content"] for completion in completions]
    
    # ground truth data
    targets = [ground_truth[i]['target'] for i in range(len(ground_truth))]
    numbers = [ground_truth[i]['numbers'] for i in range(len(ground_truth))]
    
    # extract equations from responses
    equations = [extract_solution(r) for r in responses]
    
    print(
        "-" * 20,
        f"\nQuestion:\n{prompts[0][-1]['content']}",
        f"\nTarget: {targets[0]} | Numbers: {numbers[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{equations[0]}",
    )
    
    valid_eq = [validate_equation(eqn, nums) for eqn, nums in zip(equations, numbers)]
    rewards = [0.0 for _ in range(len(responses))]
    
    for i in range(len(responses)):
        if valid_eq[i]:
            try:
                result = evaluate_equation(equations[i])
                
                if result is None:
                    continue
                
                # 2.0 correctness reward is given to responses w correct equations
                if abs(result - targets[i]) < 1e-5:
                    rewards[i] = 2.0
                else:
                    continue
            except:
                continue
    
    return rewards