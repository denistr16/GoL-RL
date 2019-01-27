import torch
import numpy as np

losses = {
    "L1_loss": torch.nn.L1Loss()
}


def sum_loss_l1(env: np.array):
    return losses["L1_loss"]((env.shape[0] * env.shape[1]), env.sum())
