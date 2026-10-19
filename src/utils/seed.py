"""
src/utils/seed.py
-----------------
Centralised random-seed utility.
Import and call set_seed(42) at the top of every training / eval script.
"""
import os
import random
import numpy as np
import torch


SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic convolutions (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] Random seed fixed to {seed}")
