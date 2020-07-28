import time
import gym
import numpy as np
from env.pong_env import PongEnv
from gym.envs.registration import register

from model import DQNSolver

register(
    id='cpong-v0',
    entry_point='env.pong_env:PongEnv'
)

if __name__ == '__main__':

    env = gym.make('cpong-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print('Run: ' + str(run) + ', exploration: ' + str(dqn_solver.exploration_rate))
                break
            dqn_solver.experience_replay()

    env.close()

    