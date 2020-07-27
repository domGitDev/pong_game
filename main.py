import time
import gym
import numpy as np
from env.pong_env import PongEnv
from gym.envs.registration import register

register(
    id='cpong-v0',
    entry_point='env.pong_env:PongEnv',
    max_episode_steps=10000,
)


if __name__ == '__main__':

    env = gym.make('cpong-v0')
    env.reset()
    for _ in range(10000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
        time.sleep(0.04)
    env.close()
    