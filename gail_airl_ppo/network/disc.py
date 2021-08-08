import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp


class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh(), mu=None):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0]+action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.mu = mu
        print("mu:", self.mu)

    def forward(self, states, actions, log_pis):
        x = torch.cat([states, actions], dim=-1)
        if self.mu is not None:
            mu = self.mu(x)
            return self.net(x) - mu * log_pis
        else:
            return self.net(x)

    def calculate_reward(self, states, actions, log_pis):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions, log_pis))

# D = - log_sigmoid(-x)
# x = log(1-e^D)
# D2 = - log_sigmoid(x - t) = - log_sigmoid(log(1-e^D) - t)

class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, gamma,
                 hidden_units_r=(100, 100),
                 hidden_units_v=(100, 100),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True),
                 mu=None,
                 state_only=False
                 ):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape[0] if state_only else state_shape[0]+action_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )
        self.mu = mu
        self.gamma = gamma
        self.state_only = state_only
        
        print("mu:", self.mu)
        print("state_only:", self.state_only)

    def f(self, states, actions, dones, next_states):
        if self.state_only:
            rs = self.g(states)
        else:
            rs = self.g(torch.cat([states, actions], dim=-1))
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, actions, dones, log_pis, next_states):
        if self.mu is not None:
            mu = self.mu(torch.cat([states, actions], dim=-1))
            log_pis = mu * log_pis
        return self.f(states, actions, dones, next_states) - log_pis
        

    def calculate_reward(self, states, actions, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, actions, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)

# D(s,a) = - log_sigmoid(log_pi - f)
# D2 = - log_sigmoid(mu * log_pi - f)
# D(s,a)/D2 = log_sigmoid(log_pi - f) / log_sigmoid(mu * log_pi - f)
# = log(1/(1+exp(f - log_pi) - 1/(1+exp(f - mu * log_pi)))
# = log()