"""
Safe array operations to prevent NumPy boolean evaluation errors
"""

import numpy as np
from typing import Any, Union, List, Optional

def safe_array_check(arr: Any) -> bool:
    """Safely check if array has values without triggering boolean ambiguity"""
    try:
        if arr is None:
            return False
        arr = np.asarray(arr)
        return arr.size > 0
    except Exception:
        return False

def safe_array_length(arr: Any) -> int:
    """Safely get array length"""
    try:
        if arr is None:
            return 0
        arr = np.asarray(arr)
        return arr.size
    except Exception:
        return 0

def safe_array_any(arr: Any) -> bool:
    """Safely check if any element is True"""
    try:
        if arr is None:
            return False
        arr = np.asarray(arr)
        if arr.size == 0:
            return False
        return np.any(arr)
    except Exception:
        return False

def safe_array_all(arr: Any) -> bool:
    """Safely check if all elements are True"""
    try:
        if arr is None:
            return False
        arr = np.asarray(arr)
        if arr.size == 0:
            return False
        return np.all(arr)
    except Exception:
        return False

def safe_array_equal(arr1: Any, arr2: Any) -> bool:
    """Safely check if arrays are equal"""
    try:
        if arr1 is None or arr2 is None:
            return arr1 is arr2
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)
        if arr1.size == 0 or arr2.size == 0:
            return arr1.size == arr2.size
        return np.allclose(arr1, arr2, rtol=1e-9, atol=1e-9)
    except Exception:
        return False

def safe_boolean_condition(condition: Any) -> bool:
    """Safely evaluate boolean condition from array"""
    try:
        if condition is None:
            return False
        if isinstance(condition, (bool, int, float)):
            return bool(condition)
        condition = np.asarray(condition)
        if condition.size == 0:
            return False
        if condition.size == 1:
            return bool(condition.item())
        # For multi-element arrays, use any() as default
        return np.any(condition)
    except Exception:
        return False

def safe_array_min_max(arr: Any) -> tuple:
    """Safely get min and max values"""
    try:
        if arr is None:
            return 0.0, 0.0
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0.0, 0.0
        return float(np.min(arr)), float(np.max(arr))
    except Exception:
        return 0.0, 0.0

def safe_array_mean(arr: Any) -> float:
    """Safely calculate array mean"""
    try:
        if arr is None:
            return 0.0
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0.0
        return float(np.mean(arr))
    except Exception:
        return 0.0