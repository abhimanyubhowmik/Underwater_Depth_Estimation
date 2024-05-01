import torch
import random
import numpy as np

def eps(x):
    """Return the `eps` value for the given `input` dtype. (default=float32 ~= 1.19e-7)"""
    dtype = torch.float32 if x is None else x.dtype
    return torch.finfo(dtype).eps

def to_log(depth):
    """Convert linear depth into log depth.
        Input: Torch Tensor"""
    depth = torch.tensor(depth)
    depth = (depth > 0) * depth.clamp(min=eps(depth)).log()
    return depth

def to_inv(depth):
    """Convert linear depth into disparity.
        Input: Torch Tensor"""
    depth = torch.tensor(depth)
    disp = (depth > 0) / depth.clamp(min=eps(depth))
    return disp

def min_max_normalize(image):
    """Min-Max normalization of Numpy array"""
# Get the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    # Normalize the image
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def fix_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.deterministic = False # speed up maybe
    torch.backends.cudnn.benchmark = True
