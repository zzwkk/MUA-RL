# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Utilities to check if packages are available.
We assume package availability won't change during runtime.
"""

import importlib.util
from functools import cache
from typing import List, Optional


@cache
def is_megatron_core_available():
    try:
        mcore_spec = importlib.util.find_spec("megatron.core")
    except ModuleNotFoundError:
        mcore_spec = None
    return mcore_spec is not None


@cache
def is_vllm_available():
    try:
        vllm_spec = importlib.util.find_spec("vllm")
    except ModuleNotFoundError:
        vllm_spec = None
    return vllm_spec is not None


@cache
def is_sglang_available():
    try:
        sglang_spec = importlib.util.find_spec("sglang")
    except ModuleNotFoundError:
        sglang_spec = None
    return sglang_spec is not None


@cache
def is_trl_available():
    try:
        trl_spec = importlib.util.find_spec("trl")
    except ModuleNotFoundError:
        trl_spec = None
    return trl_spec is not None


def import_external_libs(external_libs=None):
    if external_libs is None:
        return
    if not isinstance(external_libs, List):
        external_libs = [external_libs]
    import importlib

    for external_lib in external_libs:
        importlib.import_module(external_lib)


def load_extern_type(file_path: Optional[str], type_name: Optional[str]):
    """Load a external data type based on the file path and type name"""
    import importlib.util
    import os

    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Custom type file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}'") from e

    if not hasattr(module, type_name):
        raise AttributeError(f"Custom type '{type_name}' not found in '{file_path}'.")

    return getattr(module, type_name)


def _get_qualified_name(func):
    """Get full qualified name including module and class (if any)."""
    module = func.__module__
    qualname = func.__qualname__
    return f"{module}.{qualname}"


def deprecated(replacement: str = ""):
    """Decorator to mark APIs as deprecated."""
    import functools
    import warnings

    def decorator(func):
        qualified_name = _get_qualified_name(func)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            msg = f"Warning: API '{qualified_name}' is deprecated."
            if replacement:
                msg += f" Please use '{replacement}' instead."
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped

    return decorator
