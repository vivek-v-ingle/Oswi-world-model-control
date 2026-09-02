"""
Image and Video Preprocessing Utilities for OSVI-WM.
"""

from typing import List, Union
import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def normalize_image(img_rgb: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Resizes and normalizes an RGB image [0, 255] -> normalized float32.
    """
    if img_rgb.shape[0] != target_size[0] or img_rgb.shape[1] != target_size[1]:
        img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - MEAN) / STD
    # Transpose to channel first: (H, W, C) -> (C, H, W)
    return img_norm.transpose(2, 0, 1)


def preprocess_frame(frame: np.ndarray, repeat_frames: int = 2) -> torch.Tensor:
    """
    Prepares a single RGB observation frame into a model-ready tensor of shape (1, repeat_frames, 3, 224, 224).
    """
    norm = normalize_image(frame)
    tensor = torch.from_numpy(norm).unsqueeze(0).repeat(repeat_frames, 1, 1, 1)  # (repeat_frames, 3, H, W)
    return tensor.unsqueeze(0)  # (1, repeat_frames, 3, H, W)


def preprocess_sequence(
    frames: Union[List[np.ndarray], np.ndarray],
    target_len: int = 10,
    target_size=(224, 224),
) -> torch.Tensor:
    """
    Resamples and normalizes a sequence of RGB frames to exactly target_len frames of shape (1, target_len, 3, 224, 224).
    """
    if isinstance(frames, list):
        frames = np.array(frames)

    N = len(frames)
    if N == 0:
        raise ValueError("Frames sequence is empty.")

    if N != target_len:
        # Uniformly sample indices
        indices = np.linspace(0, N - 1, target_len, dtype=int)
        sampled_frames = frames[indices]
    else:
        sampled_frames = frames

    processed = [normalize_image(f, target_size=target_size) for f in sampled_frames]
    stacked = np.stack(processed, axis=0)  # (T, 3, H, W)
    tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, T, 3, H, W)
    return tensor
