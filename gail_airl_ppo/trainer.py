from gail_airl_ppo.network.utils import calculate_log_pi
from gail_airl_ppo.algo import GAIL, AIRL
import os
from time import time, sleep
from datetime import timedelta
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from create_heatmap import plot_heatmap

class Trainer:

    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def calculate_reward(self, states):
        with torch.no_grad():
            qpos = np.array([states[0] - 0.3, states[1]])
            qvel = np.array([0, 0])
            self.env_test.reset()
            self.env_test.set_state(qpos, qvel)
            states = torch.FloatTensor([[states[0], states[1], 0]])
            actions, log_pis = self.algo.actor.sample(states, add_noise=False)
            if isinstance(self.algo, GAIL):
                rewards = self.algo.disc.calculate_reward(states, actions, log_pis)
            elif isinstance(self.algo, AIRL):
                rewards = self.algo.disc.calculate_reward(states, actions, True, log_pis, states)
            rewards = torch.mean(rewards)
        return rewards

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()

        for step in tqdm(range(1, self.num_steps + 1)):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(os.path.join(self.model_dir, f'step{step}'))

                if self.env.env_id.startswith("PointMaze"):
                    if self.algo.weighted:
                        plot_heatmap(self.algo.mu.calculate_mu, f"{self.log_dir}/mu_{step}", self.algo.train_irl)
                    plot_heatmap(self.calculate_reward, f"{self.log_dir}/reward_{step}", self.algo.train_irl)

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            episode_return = 0.0
            done = False

            while not done:
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'\nNum steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
