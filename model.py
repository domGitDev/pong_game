import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf

class DQNSolver:

    def __init__(self, obs_shape, num_actions, learning_rate=0.01):
        self.epsilon = 1 
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay = 0.01
        self.exploring = ''

        self.action_space = num_actions
        
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=obs_shape, activation='relu'))
        self.model.add(Dense(12, activation='relu'))
        self.model.add(Dense(num_actions, activation='linear'))
        self.model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    def encode_state(self, state, n_dims):
        return state

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def act(self, env, state):
        random_number = np.random.rand()
        # 2. Explore using the Epsilon Greedy Exploration Strategy
        if random_number <= self.epsilon:
            # Explore
            action = env.action_space.sample()
            self.exploring = 'randomly'
        else:
            # Exploit best known action
            # model dims are (batch, env.observation_space.n)
            encoded = self.encode_state(state, env.observation_space.shape[0])
            encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
            predicted = self.model.predict(encoded_reshaped).flatten()
            action = np.argmax(predicted)
            self.exploring = 'predicting'
        return action

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)

    def predict(self, state):
        return self.model.predict(state)


    