import gym
import minihack
import torch
import numpy as np
from reinforce import Reinforce
from nle import nethack

MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS = MOVE_ACTIONS + (
    nethack.Command.KICK,
)

env = gym.make("MiniHack-Quest-Hard-v0", obs_crop_h = 9, obs_crop_w = 9, actions=NAVIGATE_ACTIONS, savedir = "vidoes/") #create environment

env.reset()

#define hyperparameters
hyperparameters = {"learning_rate": 0.01, "discount_factor": 0.99, "num_episodes": 1000, "PATH": "policies/test.pt"}

reinforce = Reinforce(env=env, hyperparameters=hyperparameters)

reinforce.train()
#reinforce.load_policy()

reinforce.test_policy()