import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(64, 512),
        )

        self.target = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(64, 512)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature


    def get_intristic_reward(self, next_obs):
        pred, target = self.forward(next_obs)
        return torch.nn.functional.mse_loss(pred, target)
