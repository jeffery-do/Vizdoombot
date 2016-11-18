#! /usr/bin/env python3

from vizdoom import DoomGame
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import Mode
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution

from random import choice
from time import sleep

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.engine.topology import Merge

import skimage.color, skimage.transform
import numpy as np

import signal
import sys
import os

resolution = (30, 45)
def preprocess(img):
    print("TYPE : %s" % (type(img),))
    print("SHAPE: %s" % (img.shape,))
    new_img = np.mean(img, 2)
    new_img = np.reshape(new_img, (1, 1, new_img.shape[0], new_img.shape[1]))
    print("NEW SHAPE: %s" % (new_img.shape,))
    return new_img

class RewardCalculator():
    def __init__(self):
        self.running_total = 0

    def calc_reward(self, game):

        # Assume Action Performed
        cur_reward = -5

        # Kills
        cur_killcount = game.get_game_variable(GameVariable.KILLCOUNT)
        new_kills = cur_killcount - self.prev_killcount
        if new_kills > 0:
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            print("KILLED ITTTTTTT")
            cur_reward += 1000 * new_kills

        # Health
        cur_health = game.get_game_variable(GameVariable.HEALTH)
        diff_health = cur_health - self.prev_health
        if diff_health > 0:
            cur_reward += 10 * diff_health
        elif diff_health < 0:
            cur_reward += 20 * diff_health

        # Ammo
        cur_ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        diff_ammo = cur_ammo - self.prev_ammo
        if diff_ammo > 0:
            cur_reward += 10 * diff_ammo
        elif diff_ammo < 0:
            cur_reward += 20 * diff_ammo


        # Store This State
        self.prev_killcount = cur_killcount
        self.prev_health = cur_health
        self.prev_ammo = cur_ammo

        # Return Running Total
        self.running_total += cur_reward
        return cur_reward
    def get_total_reward(self):
        return self.running_total
    def reset(self, game):
        self.prev_killcount = game.get_game_variable(GameVariable.KILLCOUNT)
        self.prev_health = game.get_game_variable(GameVariable.HEALTH)
        self.prev_ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        self.running_total = 0

FILENAME = "our_model.h5"
class NeuralNet():
    def __init__(self):
        if os.path.exists(FILENAME):
            print("Loading from %s" % FILENAME)
            self.model = load_model(FILENAME)
        else:
            print("Building from scratch")
            self.action_model = Sequential()
            self.action_model.add(Dense(8, input_dim=5, init="uniform"))

            self.image_model = Sequential()
            self.image_model.add(Convolution2D(64, 16, 16, border_mode="same", input_shape=(1, 240, 320)))
            self.image_model.add(Activation("relu"))
            self.image_model.add(Convolution2D(32, 16, 16, border_mode="same"))
            self.image_model.add(Activation("relu"))
            self.image_model.add(Flatten())
            self.image_model.add(Dense(32, activation="sigmoid", init="uniform"))

            self.model = Sequential()
            self.model.add(Merge([self.action_model, self.image_model], mode='concat', concat_axis=1))
            self.model.add(Dense(32))
            self.model.add(Dense(1))
            self.model.compile(loss="mse", optimizer="rmsprop")
    def save(self):
        print("Saving to %s" % FILENAME)
        self.model.save(FILENAME)
        return

game = DoomGame()
calc = RewardCalculator()
random_actions = False
nn = NeuralNet()

game.set_vizdoom_path("../../bin/vizdoom")
game.set_doom_game_path("../../scenarios/freedoom2.wad")
game.set_doom_scenario_path("../../scenarios/basic.wad")
game.set_doom_map("map01")
game.set_screen_resolution(ScreenResolution.RES_320X240)
game.set_screen_format(ScreenFormat.RGB24)
game.set_depth_buffer_enabled(True)
game.set_labels_buffer_enabled(True)
game.set_automap_buffer_enabled(True)

# Sets other rendering options
game.set_render_hud(False)
game.set_render_minimal_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)
game.set_render_particles(False)
game.set_render_effects_sprites(False)

# Adds buttons that will be allowed. 
game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)
game.add_available_button(Button.MOVE_FORWARD)
game.add_available_button(Button.MOVE_BACKWARD)

# Adds game variables that will be included in state.
game.add_available_game_variable(GameVariable.AMMO2)
game.add_available_game_variable(GameVariable.SELECTED_WEAPON)

# Causes episodes to finish after 200 tics (actions)
game.set_episode_timeout(100)

# Makes episodes start after 10 tics (~after raising the weapon)
game.set_episode_start_time(10)

# Makes the window appear (turned on by default)
game.set_window_visible(False)

# Turns on the sound. (turned off by default)
game.set_sound_enabled(False)

# Sets the livin reward (for each move) to -1
game.set_living_reward(-1)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game.set_mode(Mode.PLAYER)

# Initialize the game. Further configuration won't take any effect from now on.
#game.set_console_enabled(True)
game.init()

# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# 5 more combinations are naturally possible but only 3 are included for transparency when watching.
actions = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1] ]

# Run this many episodes
episodes = 1

# Sets time that will pause the engine after each action.
# Without this everything would go too fast for you to keep track of what's happening.
# 0.05 is quite arbitrary, nice to watch with my hardware setup. 
#sleep_time = 0.028
sleep_time = 0.000

print("Total Reward:", calc.get_total_reward())
for i in range(episodes):
    print("Episode #" + str(i + 1))
    game.new_episode()
    calc.reset(game)
    prev_screen_buf = None
    while not game.is_episode_finished():

        # Gets the state
        state = game.get_state()

        # Which consists of:
        n           = state.number
        vars        = state.game_variables
        screen_buf  = state.screen_buffer
        depth_buf   = state.depth_buffer
        labels_buf  = state.labels_buffer
        automap_buf = state.automap_buffer
        labels      = state.labels

        # Makes a random action and get remember reward.
        action_set = None
        if not random_actions:
            if prev_screen_buf != None:
                print("Predicting")
                predictions = []
                for a in actions:
                    np_a = np.array([a])
                    prediction = nn.model.predict([np_a, prev_screen_buf], batch_size=1)
                    predictions.append(prediction[0])
                print("Predictions: %s" % predictions)
                max_prediction = max(predictions)
                best_actions = []
                for a,p in zip(actions, predictions):
                    if p == max_prediction:
                        best_actions.append(a)
                print("Max Prediction: %d" % max_prediction)
                print("Best Actions: %s" % best_actions)
                action_set = choice(best_actions)
                print("Previous  Reward : %.3f" % r)
                print("Predicted Reward : %.3f" % max_prediction)
                print("Difference Reward: %.3f" % (r - max_prediction))
            else:
                print("Randomly Choosing")
                action_set = choice(actions)
        else:
            action_set = choice(actions)
        # DO THE THING
        try:
            game.make_action(action_set)
        except Exception as e:
            nn.save()
            sys.exit(0)
        r = calc.calc_reward(game)

        # Makes a "prolonged" action and skip frames:
        # skiprate = 4
        # r = game.make_action(choice(actions), skiprate)

        # The same could be achieved with:
        # game.set_action(choice(actions))
        # game.advance_action(skiprate)
        # r = game.get_last_reward()

        # Prints state's game variables and reward.
        prev_screen_buf = preprocess(screen_buf)
        np_action_set = np.array([action_set])
        #np_action_set = np.reshape(np_action_set, (5,))
        nn.model.fit([np_action_set, prev_screen_buf],np.array([r]),batch_size=1, verbose=0)
        print("Labels: %s" % list(map(lambda x : x.object_name, labels)))
        print("Labels: %s" % list(map(lambda x : x.value, labels)))
        print("State %s" % state.__dict__)
        print("State #" + str(n))
        print("Game variables:", vars)
        print("Reward:", r)
        print("Total Reward:", calc.get_total_reward())
        print("KILLCOUNT: %s" % game.get_game_variable(GameVariable.KILLCOUNT))
        print("=====================")

        if sleep_time > 0:
            sleep(sleep_time)

    # Check how the episode went.
    print("Episode finished.")
    print("Total reward:", calc.get_total_reward())
    print("************************")

# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
game.close()
nn.save()
