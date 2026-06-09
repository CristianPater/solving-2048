import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.network(x)

    def get_weights(self):
        return parameters_to_vector(self.parameters()).detach().numpy()

    def set_weights(self, chromosome):
        vector_to_parameters(torch.tensor(chromosome, dtype=torch.float32), self.parameters())

    def act(self, board, valid_actions):
        x = torch.tensor(board, dtype=torch.float32)
        with torch.no_grad():
            scores = self.forward(x)
        scores = scores.numpy()
        scores[~valid_actions] = -np.inf
        return np.argmax(scores)

    def predict(self, board, env):
        valid_actions = np.array([env.valid(a) for a in [env.UP, env.DOWN, env.LEFT, env.RIGHT]])
        board = np.where(board == 0, 0, np.log2(np.maximum(board, 1))).flatten()
        return self.act(board, valid_actions)
