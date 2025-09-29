# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

# Get PR title from environment
pr_title = os.environ.get("PR_TITLE", "").strip()

# Define rules
allowed_modules = ["fsdp", "megatron", "sglang", "vllm", "rollout", "trainer"]
allowed_modules += ["tests", "training_utils", "recipe", "hardware", "deployment"]
allowed_modules += ["ray", "worker", "single_controller", "misc", "docker", "ci"]
allowed_modules += ["perf", "model", "algo", "env", "tool", "ckpt", "doc", "data"]
allowed_types = ["feat", "fix", "refactor", "chore"]

# Build dynamic regex pattern
re_modules_pattern = re.compile(r"^\[([a-z_,\s]+)\]", re.IGNORECASE)
re_modules = re_modules_pattern.match(pr_title)
if not re_modules:
    print(f"❌ Invalid PR title: '{pr_title}'")
    print("Expected format: [module] type: description")
    print(f"Allowed modules: {', '.join(allowed_modules)}")
    raise Exception("Invalid PR title")
else:
    modules = re.findall(r"[a-z]+", re_modules.group(1).lower())
    if not all(module in allowed_modules for module in modules):
        invalid_modules = [module for module in modules if module not in allowed_modules]
        print(f"❌ Invalid modules: {', '.join(invalid_modules)}")
        print(f"Allowed modules: {', '.join(allowed_modules)}")
        raise Exception("Invalid PR title")

types_pattern = "|".join(re.escape(t) for t in allowed_types)
re_types_pattern = re.compile(rf"^\[[a-z_,\s]+\]\s+({types_pattern}):\s+.+$", re.IGNORECASE)
match = re_types_pattern.match(pr_title)

if not match:
    print(f"❌ Invalid PR title: '{pr_title}'")
    print("Expected format: [module] type: description")
    print(f"Allowed types: {', '.join(allowed_types)}")
    raise Exception("Invalid PR title")

change_type = match.group(1).lower()

print(f"✅ PR title is valid: {pr_title}, modules: {modules}, type: {change_type}")
