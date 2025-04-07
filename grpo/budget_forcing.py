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
from transformers import PreTrainedTokenizerBase

import itertools


class WaitLogitsProcessor:
    """
    This class stores a set where each element is a sequence of 3 integers. If the last three tokens of the generated sequence matches any element,
    then modify the logits for the next token prediction so that all the mass concentrates on self.next_token_id.
    
    
    Each element in the set represents the "</think>" token (the end-of-thinking token for GRPO), and the 2nd integer always
    corresponds to "think". The first and third integers corresponds to all possible tokens in the
    vocabulary that ends with "</" and starts with ">", respectively.
    
    Args:
        tokenizer: PreTrainedTokenizerBase
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: torch.device, next_token_id: int, min_num_tokens: int, boost_logit: float = 1e9, low_logit: float = -1e9):
        self.tokenizer = tokenizer
        self.device = device
        
        think_token_id = 26865 # this is the "think" token for Qwen2.5-1.5B-Instruct
        # think_token_id = torch.Tensor(tokenizer("think").input_ids).long().to(self.device)
        
        vocab = tokenizer.get_vocab()        
        left_matching_token_ids = [token_id for token, token_id in vocab.items() if token.endswith("</")]
        right_matching_token_ids = [token_id for token, token_id in vocab.items() if token.startswith(">")]
        
        self.think_token_combo = torch.stack([torch.Tensor([left_token_id, think_token_id, right_token_id]) for left_token_id, right_token_id in itertools.product(left_matching_token_ids, right_matching_token_ids)], dim=0).long().to(self.device)
        # self.think_token_combo has shape (M, 3)
        
        self.next_token_id = next_token_id
        self.min_num_tokens = min_num_tokens
        self.boost_logit = boost_logit
        self.low_logit = low_logit
        
    def __call__(self, *args):
        """
        Supports two possible signatures:
          - (prompt_tokens, past_tokens, logits)
          - (past_tokens, logits)
        """
        if len(args) == 3:
            prompt_tokens, past_tokens, logits = args
        elif len(args) == 2:
            past_tokens, logits = args
        else:
            raise ValueError("Expected 2 or 3 arguments, got %d" % len(args))
        
        # print("====== Matching ======")

        # Check if the the last three tokens in past_tokens match with "</think>".
        # Assuming past_tokens is a list or tensor of token ids.
        if len(past_tokens) > 3 and len(past_tokens) < self.min_num_tokens:
            if isinstance(past_tokens, torch.Tensor):
                # If it's a tensor, get the last three elements.
                last_three_tokens = past_tokens[-3:]
            else:
                # Otherwise assume it's a list-like object.
                last_three_tokens = torch.Tensor(past_tokens[-3:]).long().to(self.device)
        
            # First reshape last_three_tokens into (1, 3).
            # Then compare with self.think_token_combo (which has shape (M, 3))
            # The result would be (M, 3)
            matches = (last_three_tokens.unsqueeze(0) == self.think_token_combo)  # shape (M, 3)
            # 3. For a complete match, all three elements must match
            pattern_match = matches.all(dim=-1)  # shape (M,)

            if pattern_match.any(dim=0).item():
                print("================ MATCHED ================")
                print(last_three_tokens)
                logits = logits.clone()  # avoid in-place modification if needed
                logits.fill_(self.low_logit)
                logits[self.next_token_id] = self.boost_logit

        return logits
