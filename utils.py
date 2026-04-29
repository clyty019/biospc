#!/usr/bin/env python
# coding: utf-8
"""Common utility functions"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set the random seed for torch, numpy, and random to ensure result consistency"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
