import numpy as np
import torch

class RandomAgent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def get_action_probs(self, *args):
        return torch.sigmoid(torch.randn(self.output_size))

    def reflect(self, *args):
        return None
