import torch
from torch import nn

from .utils import build_mlp, reparameterize, evaluate_lop_pi


class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        actions, log_pis = reparameterize(self.net(states), self.log_stds)
        return actions, log_pis

    def evaluate_log_pi(self, states, actions):
        log_pis = evaluate_lop_pi(self.net(states), self.log_stds, actions)
        return log_pis


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        log_stds = torch.clamp(log_stds, -20, 2)
        actions, log_pis = reparameterize(means, log_stds)
        return actions, log_pis

    def evaluate_log_pi(self, states, actions):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        log_stds = torch.clamp(log_stds, -20, 2)
        log_pis = evaluate_lop_pi(means, log_stds, actions)
        return log_pis