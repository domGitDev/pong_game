import time
import gym
import numpy as np
from env.pong_env import PongEnv
from gym.envs.registration import register

from model import DQNSolver

register(
    id='cpong-v0',
    entry_point='env.pong_env:PongEnv',
    max_episode_steps=10000,
)


def train_pong():
    env = gym.make('cpong-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    while True:
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        while True:
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            dqn_solver.experience_replay()
            state = state_next
            if terminal:
                break

    env.close()


if __name__ == '__main__':

    train_pong()

    '''
    env = gym.make('cpong-v0')
    env.reset()
    for _ in range(10000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
        time.sleep(0.04)
    env.close()
    '''
    