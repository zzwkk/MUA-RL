import random
import re
import string
import logging
import os
import json
from hashlib import sha256
from typing import Dict, List, Set, Tuple, Union
# from verl.tools.taubench_retail.data import load_data
# from verl.workers.rollout.sglang_rollout.sglang_rollout_taubench import terminate_tools

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ToHashable = Union[
    str, int, float, Dict[str, "ToHashable"], List["ToHashable"], Set["ToHashable"]
]
Hashable = Union[str, int, float, Tuple["Hashable"], Tuple[Tuple[str, "Hashable"]]]


def to_hashable(item: ToHashable) -> Hashable:
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item


def consistent_hash(
    value: Hashable,
) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()


def get_data_hash(data) -> str:
    return consistent_hash(to_hashable(data))

def clean_arg(action):
    """清理为None的参数"""
    new_action = {
        'name': action['name'],
        'arguments': {k: v for k, v in action['arguments'].items() if v is not None}
    }
    return new_action

def compute_score(taubench_database, messages, solution_str, ground_truth, extra_info) -> float:
    """
        Compute reward score based on database state comparison.
    
    Args:
        taubench_database: Shared database state from rollout request
        messages: Conversation messages
        solution_str: Generated solution string
        ground_truth: Ground truth data containing tool calls
        extra_info: Additional information
        
    Returns:
        Computed reward score
        
    Logic:
        - Extract shared data from rollout request
        - Extract all tool_calls from ground_truth and execute them on the database
        - Compare hash values of both databases
        - If hash values are identical, return the reward score
    """
    data_hash = get_data_hash(taubench_database) 
    messages = [msg.model_dump() for msg in messages['messages'] if msg.role == 'assistant' and msg.tool_calls is None] 
    reward = 1.0
    actions = [clean_arg(action) for action in ground_truth["actions"]] # correct traj[{'name':'tool_name', 'arguments':{xxx}}, {...}]
    task_outputs = ground_truth['outputs']
    
    if len(task_outputs) > 0:
        # check outputs
        r_outputs = 1.0
        outputs = {}
        for output in task_outputs:
            found = False
            for message in messages:
                if output.lower() in message['content'].lower().replace(",", ""):
                    found = True
                    break
            outputs[output] = found
            if not found:
                r_outputs = 0.0
                reward = 0.0
            
    # check database change
    gt_data_hash = ground_truth["gt_data_hash"]
    if not data_hash == gt_data_hash:
        reward = 0.0

    print(f"========================================\n")
    print(f"\n[ID: {extra_info['index']}]")
    print(f"\n[ID: {extra_info['index']}][Raw Str]\n{repr(solution_str)}")
    print(f"\n[ID: {extra_info['index']}][Gold Traj]\n{actions}")
    print(f"\n[ID: {extra_info['index']}][Gold Hash]\n{gt_data_hash}")
    print(f"\n[ID: {extra_info['index']}][Trajectory Hash]\n{data_hash}")
    print(f"\n[ID: {extra_info['index']}][Reward Score]\n{reward}")
    print(f"========================================\n")

    return reward
    