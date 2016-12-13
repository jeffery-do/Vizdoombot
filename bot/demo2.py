#! /usr/bin/env python3


from vizdoom import Button
from vizdoom import GameVariable

import itertools as it
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

import numpy as np

import sys
import os
import time

from bot.reward import RewardCalculator
from bot import doomgame
from bot.image import preprocess
from bot.memory import ReplayMemory

FILENAME = "our_model.h5"
resolution = (30, 45)

class NeuralNet():
    def __init__(self, gamma, output_dim):
        self.gamma = gamma
        self.output_dim = output_dim
        if os.path.exists(FILENAME):
            print("Loading from %s" % FILENAME)
            self.model = load_model(FILENAME)
        else:
            print("Building from scratch")
            self.model = Sequential()
            self.model.add(Convolution2D(32, 7, 7, border_mode="same", input_shape=(30, 45, 3)))
            self.model.add(Activation("sigmoid"))
            self.model.add(MaxPooling2D())
            self.model.add(Convolution2D(32, 4, 4, border_mode="same"))
            self.model.add(Activation("sigmoid"))
            self.model.add(MaxPooling2D())
            self.model.add(Flatten())
            self.model.add(Dense(800))
            self.model.add(Activation("sigmoid"))
            self.model.add(Dense(output_dim))
            self.model.compile(loss="mse", optimizer="rmsprop")

    def get_learn_val(self, s1, s2, a, reward, terminal):
        qvals = nn.model.predict(s1.reshape((1,30,45,3)), batch_size=1)
        y = np.zeros(self.output_dim)
        y[:] = qvals[:]
        if not terminal:
            new_qvals = nn.model.predict([s2], batch_size=1)
            max_q = np.max(new_qvals)
            update = reward + (self.gamma * max_q)
        else:
            update = reward
        y[a] = update
        return y

    def learn_batch(self, s1_batch, s2_batch, a_batch, reward_batch, terminal_batch):
        tuples = []
        for i in range(len(s1_batch)):
            tuples.append((s1_batch[i], s2_batch[i], a_batch[i], reward_batch[i], terminal_batch[i]))

        correct_results = np.array(list(map(lambda x: self.get_learn_val(*x), tuples)))
        print("shape s1 batch", s1_batch.shape)
        self.model.train_on_batch(s1_batch, correct_results)

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
    Button.ATTACK
]

batch_size = 64
episodes = 100000

def create_actions(n):
    actions = sorted(list(it.product([True, False], repeat=n)))
    print(actions)
    return actions

game = doomgame.init(buttons)
num_buttons = game.get_available_buttons_size()
actions = create_actions(num_buttons)
memory = ReplayMemory(10000, resolution)
nn = NeuralNet(gamma, len(actions))

print("Total Reward:", calc.get_total_reward())
num_kills = 0

allow_random = True
scores = []

for i in range(episodes):
    epsilon = 0.2
    print("Episode #" + str(i + 1))
    game.new_episode()
    calc.reset(game)
    # Gets the state
    state = game.get_state()
    while not game.is_episode_finished():
        time.sleep(0.028)
        # Which consists of:
        screen_buf = preprocess(state.screen_buffer, resolution).reshape((1, resolution[0], resolution[1], 3))
        # Guess Q
        if allow_random and np.random.rand() < epsilon:
            action_ndx = np.random.randint(0, len(actions))
            print("RANDOM ACTION")
        else:
            qvals = nn.model.predict(screen_buf, batch_size=1)
            action_ndx = (np.argmax(qvals))
            print("CHOSEN ACTION")

        # Perform Action
        try:
            print(actions[action_ndx])
            game.make_action(list(actions[action_ndx]), 10)
        except Exception as e:
            print(e)
            nn.save()
            sys.exit(0)

        reward = calc.calc_reward(game)
        if not game.is_episode_finished():
            new_state = game.get_state()
            new_screen_buf = preprocess(new_state.screen_buffer, resolution).reshape((1, resolution[0], resolution[1], 3))
            memory.add_transition(screen_buf, new_screen_buf, action_ndx, reward, False)
            state = new_state
        else:
            memory.add_transition(screen_buf, None, action_ndx, reward, True)

        # Prints state's game variables and reward.
        print("State %s" % state.__dict__)
        print("Total Reward:", calc.get_total_reward())
        print("Num Kills:", num_kills)
        print("Episode number:", i)
        print("Random move probability: ", epsilon)
        print("=====================")
    if game.get_game_variable(GameVariable.KILLCOUNT) > 0:
        num_kills += 1
    nn.learn_from_memory(memory, batch_size=batch_size)
    # Check how the episode went.
    print("Episode finished.")
    print("Num Kills:", num_kills)
    total_score = calc.get_total_reward()
    print("Total reward:", total_score)
    print("************************")
    scores.append(total_score)

# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
game.close()
nn.save()

def save_scores(scores):
    with open("try%s.csv" % time.time(), "w") as f:
        for s in scores:
            print(s, file=f)

save_scores(scores)