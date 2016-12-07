#! /usr/bin/env python3


from vizdoom import Button
from vizdoom import GameVariable

import itertools as it
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

import numpy as np

import sys
import os

from bot.reward import RewardCalculator
from bot import doomgame
from bot.image import preprocess
from bot.memory import ReplayMemory

FILENAME = "our_model.h5"
resolution = (30, 45)

class NeuralNet():
    def __init__(self, gamma, output_dim):
        self.gamma = gamma
        if os.path.exists(FILENAME):
            print("Loading from %s" % FILENAME)
            self.model = load_model(FILENAME)
        else:
            print("Building from scratch")
            self.model = Sequential()
            self.model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(30, 45, 3)))
            self.model.add(Activation("relu"))
            self.model.add(Flatten())
            self.model.add(Dense(32, activation="sigmoid", init="uniform"))
            self.model.add(Dense(output_dim))
            self.model.compile(loss="mse", optimizer="rmsprop")

    def get_learn_val(self, s1, s2, a, reward, terminal):
        qvals = nn.model.predict([s1], batch_size=1)
        y = np.zeros((1, 3))
        y[:] = qvals[:]
        if not terminal:
            new_qvals = nn.model.predict([s2], batch_size=1)
            max_q = np.max(new_qvals)
            update = reward + (self.gamma * max_q)
        else:
            update = reward
        y[0][action_ndx] = update
        return y

    def learn_batch(self, s1_batch, s2_batch, a_batch, reward_batch, terminal_batch):
        tuples = []
        for i in range(len(s1_batch)):
            tuples.append((s1_batch[i], s2_batch[i], a_batch[i], reward_batch[i], terminal_batch[i]))

        correct_results = list(map(lambda s1, s2, a, reward, terminal: self.get_learn_val(s1, s2, a, reward, terminal), tuples))
        self.model.fit(s1_batch, correct_results, batch_size=len(correct_results))

    def learn_from_memory(self, memory, batch_size):
        if memory.size > batch_size:
            s1, a, s2, isterminal, r = memory.get_sample(batch_size)
            self.learn_batch(s1, s2, a, r, isterminal)

    def save(self):
        print("Saving to %s" % FILENAME)
        self.model.save(FILENAME)

calc = RewardCalculator()
gamma = 0.99

buttons = [
    Button.MOVE_LEFT,
    Button.MOVE_RIGHT,
    Button.ATTACK,
    Button.MOVE_FORWARD,
    Button.MOVE_BACKWARD
]

batch_size = 64
episodes = 5000
epsilon = 1.0

game = doomgame.init(buttons)
num_actions = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]
memory = ReplayMemory(10000, resolution)
nn = NeuralNet(gamma, num_actions)

print("Total Reward:", calc.get_total_reward())
num_kills = 0



for i in range(episodes):
    if i % 1000 == 0:
        epsilon -= 0.05
    print("Episode #" + str(i + 1))
    game.new_episode()
    calc.reset(game)
    # Gets the state
    state = game.get_state()
    while not game.is_episode_finished():
        # Which consists of:
        screen_buf = preprocess(state.screen_buffer, resolution)
        # Guess Q
        qvals = nn.model.predict([screen_buf], batch_size=1)
        if np.random.rand() < epsilon:
            action_ndx = np.random.randint(0, num_actions)
        else:
            action_ndx = (np.argmax(qvals))

        # Perform Action
        try:
            game.make_action(actions[action_ndx])
        except Exception as e:
            nn.save()
            sys.exit(0)

        reward = calc.calc_reward(game)
        if not game.is_episode_finished():
            new_state = game.get_state()
            new_screen_buf = preprocess(new_state.screen_buffer, resolution)
            memory.add_transition(screen_buf, new_screen_buf, action_ndx, reward, False)
            state = new_state
        else:
            memory.add_transition(screen_buf, None, action_ndx, reward, True)

        nn.learn_from_memory(memory, batch_size=64)

        # Prints state's game variables and reward.
        print("State %s" % state.__dict__)
        print("Action:", ["Left","Right","Shoot"][action_ndx])
        print("Total Reward:", calc.get_total_reward())
        print("KILLCOUNT: %s" % game.get_game_variable(GameVariable.KILLCOUNT))
        num_kills = game.get_game_variable(GameVariable.KILLCOUNT)
        print("=====================")
    # Check how the episode went.
    print("Episode finished.")
    print("Num Kills:", num_kills)
    print("Total reward:", calc.get_total_reward())
    print("************************")

# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
game.close()
nn.save()
