"""
A Minimal Deep Q-Learning Implementation (minDQN)
Running this code will render the agent solving the CartPole environment 
using OpenAI gym. Our Minimal Deep Q-Network is approximately 150 lines of code. 
In addition, this implementation uses Tensorflow and Keras and should generally 
run in less than 15 minutes.

Usage: python3 minDQN.py
"""
import os
import time
import random
import argparse
import numpy as np
from collections import deque

import gym
import tensorflow as tf
from gym import wrappers
from gym.envs.registration import register
from gymenv.pong_env import PongEnv

from model import DQNSolver


register(
    id='cpong-v0',
    entry_point='gymenv.pong_env:PongEnv'
)

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

env = gym.make('cpong-v0')
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# An episode a full game
train_episodes = 1000
test_episodes = 300


def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([encode_observation(batch[0],env.observation_space.shape) for batch in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([encode_observation(batch[3], env.observation_space.shape) for batch in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, _, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(encode_observation(observation, env.observation_space.shape))
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

def encode_observation(observation, n_dims):
    return observation


def main(args):

    if args.train:
        epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
        max_epsilon = 1 # You can't explore more than 100% of the time
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
        decay = 0.01

        # 1. Initialize the Target and Main models
        # Main Model (updated every step)
        model = DQNSolver(env.observation_space.shape, env.action_space.n)
        # Target Model (updated every 100 steps)
        target_model = DQNSolver(env.observation_space.shape, env.action_space.n)
        target_model.set_weights(model.get_weights())

        replay_memory = deque(maxlen=60_000)
        steps_to_update_target_model = 0
        
        for episode in range(train_episodes):
            total_training_rewards = 0
            observation = env.reset()
            done = False
            while not done:
                steps_to_update_target_model += 1
                action = model.act(env, observation)

                new_observation, reward, done, info = env.step(action)
                #if reward or done:
                replay_memory.append([observation, action, reward, new_observation, done])

                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 4 == 0 or done:
                    train(env, replay_memory, model, target_model, done)

                observation = new_observation
                if reward > 0:
                    total_training_rewards += reward

                if done:
                    print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                    if reward > 0:
                        total_training_rewards += 1

                    if steps_to_update_target_model >= 1000:
                        print('Copying main network weights to the target network weights')
                        target_model.set_weights(model.get_weights())
                        steps_to_update_target_model = 0
                    break
                time.sleep(0.001)
                env.render()

            print(model.exploring)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
            model.set_epsilon(epsilon)
        
        # save trained target model
        target_model.save_weights(args.weights)

    else:
        print('TESTING...')
        
        max_rewards = 0
        env.monitor.start(os.path.join(args.logs, 'cartpole'), force=True)
        target_model = DQNSolver(env.observation_space.shape, env.action_space.n)
        target_model.load_weights(args.weights)

        for episode in range(test_episodes):
            rewards = 0
            steps = 0
            done = False

            env.reset()
            action = env.action_space.sample()
            new_observation, reward, done, info = env.step(action)
            observation = new_observation

            while not done:
                env.render()
                action = model.predict(observation, env.observation_space.shape[0])
                
                new_observation, reward, done, info = env.step(action)
                observation = new_observation

                steps += 1
                rewards += reward
                max_rewards = max(max_rewards, rewards)

                if done:
                    break

            print("Testing steps: {} rewards {} max {}: ".format(steps, rewards, max_rewards))

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pong Game.')
    parser.add_argument('--train', default=False, type=bool,
                        help="train agent")
    parser.add_argument('--weights', type=str,
                        default="./logs/cartpole_weights.h5", 
                        help="test agent")

    args = parser.parse_args()
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    main(args)