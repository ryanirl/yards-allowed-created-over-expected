import numpy as np
import logging
import torch
import sys

from typing import Tuple
from torch import Tensor

X_MID = 120.0 / 2
Y_MID =  53.3 / 2


def circular_encode(degrees: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    radians = np.radians(degrees)
    return np.sin(radians), np.cos(radians)


def circular_decode(dir_x: np.ndarray, dir_y: np.ndarray) -> np.ndarray:
    radians = np.arctan2(dir_x, dir_y)
    degrees = np.degrees(radians) % 360
    return degrees


def get_velocity(x: np.ndarray, y: np.ndarray, axis: int = 0) -> np.ndarray:
    vx = np.diff(x, axis = axis)
    vy = np.diff(y, axis = axis)

    # Maintain original shape
    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    return np.sqrt(vx ** 2 + vy ** 2) * 9.8 # Linear regression was giving me 9.8


def convert_output(output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = (output[:, :, 0] * X_MID) + X_MID 
    y = (output[:, :, 1] * Y_MID) + Y_MID
    d = circular_decode(output[:, :, 2], output[:, :, 3])
    v = get_velocity(x, y)

    return x, y, d, v


def prepare_data(data: np.ndarray) -> np.ndarray:
    # data: (seq_len, n_players, 9)
    # ["x", "y", "s", "a", "dis", "o", "dir", "isDef", "isBallCarrier"]
    seq_len, n_players, _ = data.shape

    x = (data[:, :, 0].copy() - X_MID) / X_MID
    y = (data[:, :, 1].copy() - Y_MID) / Y_MID

    is_def = data[:, :, 7]
    is_bc  = data[:, :, 8]
    is_def = np.where(is_def > 0.5, 1, -1)
    is_bc  = np.where(is_bc  > 0.5, 1, 0)

    dir_x, dir_y = circular_encode(data[:, :, 6])

    x_diff = np.diff(x, prepend = 0, axis = 0)
    y_diff = np.diff(y, prepend = 0, axis = 0)
    x_diff[0] = 0
    y_diff[0] = 0

    dis = np.zeros((seq_len, n_players))
    x_rel = np.zeros((seq_len, n_players))
    y_rel = np.zeros((seq_len, n_players))
    for t in range(seq_len):
        bc_inds = data[t, :, -1] > 0.5
        if np.any(bc_inds):
            x_rel[t] = x[t, bc_inds] - x[t]
            y_rel[t] = y[t, bc_inds] - y[t]
            dis[t] = np.sqrt(x_rel[t] ** 2 + y_rel[t] ** 2)

    out = np.stack([x, y, dir_x, dir_y, dis, x_rel, y_rel, x_diff, y_diff, is_def, is_bc])
    out = out.transpose(1, 2, 0)

    return out


@torch.no_grad()
def combine(prev: Tensor, trajectory: Tensor, direction: Tensor) -> Tensor:
    prev       = prev[:, -1].unsqueeze(1)       # b, l, n, c 
    direction  = direction[:, -1].unsqueeze(1)  # b, l, n, 2
    trajectory = trajectory[:, -1].unsqueeze(1) # b, l, n, 2

    # Every next-step prediction is a function of the previous. 
    out = prev.clone() 

    # The first two features are the XY. The model learns the offset and so 
    # we add the offset to obtain the new XY.
    out[:, :, :, 0:2] = trajectory
    out[:, :, :, 2:4] = direction
    
    # Player-specific features.
    for i in range(out.shape[0]):
        curr = out[i].clone()

        # XY position features
        x = curr[:, :, 0].clone()
        y = curr[:, :, 1].clone()

        # XY relative to ball carrier. Also handle the case in which there 
        # is no known ball carrier yet (before play start).
        bc_inds = curr[:, :, -1] > 0.5
        if not torch.any(bc_inds): 
            x_rel = torch.zeros_like(x)
            y_rel = torch.zeros_like(x)
            dis = torch.zeros_like(x)
        else:
            x_rel = x[bc_inds] - x.clone() 
            y_rel = y[bc_inds] - y.clone()
            dis = torch.sqrt(x_rel ** 2 + y_rel ** 2)

        # Add the features to the output 
        out[i, :, :, 4] = dis   # Distance to ball carrier 
        out[i, :, :, 5] = x_rel # X relative to ball carrier
        out[i, :, :, 6] = y_rel # Y relative to ball carrier
    
    out[:, :, :, 7] = out[:, :, :, 0] - prev[:, :, :, 0] # curr_x_diff
    out[:, :, :, 8] = out[:, :, :, 1] - prev[:, :, :, 1] # curr_y_diff 
        
    return out



