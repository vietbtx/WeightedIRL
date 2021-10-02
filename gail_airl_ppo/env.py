import gym
import mujoco_py
import airl_envs
gym.logger.set_level(40)


def make_env(env_id):
    return NormalizedEnv(gym.make(env_id), env_id)


class NormalizedEnv(gym.Wrapper):

    def __init__(self, env, env_id):
        gym.Wrapper.__init__(self, env)
        try:
            self._max_episode_steps = env._max_episode_steps
        except:
            self._max_episode_steps = 500
        self.scale = env.action_space.high
        self.action_space.high /= self.scale
        self.action_space.low /= self.scale
        self.env_id = env_id

    def step(self, action):
        return self.env.step(action * self.scale)
