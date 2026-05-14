from __future__ import annotations

import random


def split_lines(lines: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    random.seed(seed)
    lines = lines[:]
    random.shuffle(lines)
    n = len(lines)
    n_train = int(n * 0.9)
    n_val = int(n * 0.05)
    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n - n_train - n_val)
        n_train = max(1, n - n_val - n_test)
    else:
        n_test = n - n_train - n_val
    train = lines[:n_train]
    val = lines[n_train : n_train + n_val]
    test = lines[n_train + n_val : n_train + n_val + n_test]
    return train, val, test
