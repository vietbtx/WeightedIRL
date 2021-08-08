import torch
from torch import nn
from torch.optim import Adam

from .base import Algorithm
from gail_airl_ppo.buffer import RolloutBuffer
from gail_airl_ppo.network import StateIndependentPolicy, StateFunction
from gail_airl_ppo.network.utils import build_mlp


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0, weighted=False, state_only=False):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        self.units_actor = units_actor
        self.units_critic = units_critic
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.state_only = state_only
        self.weighted = weighted
        self.mix_buffer = mix_buffer
        self.reset()

    def reset(self):
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_length,
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            device=self.device,
            mix=self.mix_buffer
        )
    
        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=self.units_actor,
            hidden_activation=nn.Tanh()
        ).to(self.device)

        # Critic.
        self.critic = StateFunction(
            state_shape=self.state_shape,
            hidden_units=self.units_critic,
            hidden_activation=nn.Tanh()
        ).to(self.device)

        self.optim_actor = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=self.lr_critic)
        if self.weighted:
            self.mu = build_mlp(
                input_dim=self.state_shape[0] + self.action_shape[0],
                output_dim=self.action_shape[0],
                hidden_units=(100, 100),
                hidden_activation=nn.ReLU(inplace=True)
            )
        else:
            self.mu = None

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1
        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states, writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar('loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        if self.mu:
            mu = self.mu(torch.cat([states, actions], dim=-1))
            entropy = - (mu * log_pis).mean()
        else:
            entropy = - log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar('loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar('stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        pass
