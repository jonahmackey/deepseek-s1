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

from datasets import Dataset, load_dataset

def get_dataset(task="gsm8k", split="train", num_examples=-1) -> Dataset:
    if task == "gsm8k":
        return get_gsm8k_questions(split, num_examples)
    elif task == "countdown":
        return get_countdown_questions(split, num_examples)
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks are 'gsm8k' and 'countdown'.")

# GSM8K
SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train", num_examples=-1) -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    
    if num_examples > 0:
        data = data.select(range(min(num_examples, len(data))))
        
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


# Countdown
def make_prefix(dp, template_type):
    target = dp['target']
    numbers = dp['nums']
    
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

def process_fn(example):
    question = make_prefix(example, template_type='qwen-instruct')
    solution = {
        "target": example['target'],
        "numbers": example['nums']
    }
    data = {
        "prompt": [
            {"role": "user", "content": question},
            ],
        "ground_truth": solution,
    }
    return data

def get_countdown_questions(split="train", num_examples=-1) -> Dataset:
    train_size = 100000
    test_size = 1024
    
    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
    
    if split == "train":
        idx1 = 0
        idx2 = min(num_examples, train_size) if num_examples > 0 else train_size
    elif split == "test":
        idx1 = train_size
        idx2 = train_size + min(num_examples, test_size) if num_examples > 0 else test_size
    
    dataset = raw_dataset.select(range(idx1, idx2))    
    dataset = dataset.map(function=process_fn)
    
    return dataset
        
    