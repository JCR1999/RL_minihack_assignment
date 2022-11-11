from os import stat
import random
import numpy as np
from nle import nethack
import gym
import minihack
import cv2
from starter_code.agent import DQNAgent
from starter_code.replay_buffer import ReplayBuffer
from starter_code.wrappers import *
import torch.nn.functional as F
import torch
cv2.ocl.setUseOpenCL(False)
from minihack import RewardManager
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import cv2
cv2.ocl.setUseOpenCL(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pixel_crop_rows = 9
pixel_crop_cols = 9

class ColorPixelRenderingReinforce(gym.Wrapper):
    def __init__(self, env, key_name="pixel"):
        super().__init__(env)
        self.last_pixels = None
        self.viewer = None
        self.key_name = key_name

        render_modes = env.metadata['render.modes']
        render_modes.append("rgb_array")
        env.metadata['render.modes'] = render_modes

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_pixels = obs[self.key_name]
        return obs, reward, done, info

    def render(self, mode="human", **kwargs):
        img = self.last_pixels

        # Hacky but works
        if mode != "human":
            return img
        else:
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def reset(self):
        obs = self.env.reset()
        self.last_pixels = obs[self.key_name]
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def print2dState(s):
    for i in range(9):
        for j in range(9):
            val = s[i*9 + j] 
            if val<10:
                print(val, "    ", end="")
            elif val<100:
                print(val, "   ", end="")
            elif val<1000:
                print(val, "  ", end="")
            else:
                print(val, " ", end="")
        print()
    print()

class ColorPixelRenderingDQN(gym.Wrapper):
    def __init__(self, env, input_name):
        super().__init__(env)
        self.viewer = None
        self.last_pixels = None
        self.key_name = input_name
        rm = env.metadata['render.modes']
        rm.append("rgb_array")
        env.metadata['render.modes'] = rm

    def reset(self):
        s = self.env.reset()
        self.last_pixels = s[self.key_name]
        return s

    def render(self, **kwargs):
        return self.last_pixels

    def step(self, action):
        s, r, done, info = self.env.step(action)
        self.last_pixels = s[self.key_name]
        return s, r, done, info

def averageRewards(average_rewards):
    average_rewards_list = []
    for episode_num in average_rewards.keys():
        episode_rewards = average_rewards[episode_num]
        average_rewards_list.append(sum(episode_rewards)/len(episode_rewards))
    return average_rewards_list


if __name__ == "__main__":
    #MOVE_ACTIONS corresponds to the 8 compass directions
    MOVE_ACTIONS = tuple(nethack.CompassDirection)

    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
        nethack.Command.OPEN,
        nethack.Command.QUAFF,
        nethack.Command.ZAP,
        nethack.Command.INVOKE,
        nethack.Command.WEAR,   
        nethack.Command.WIELD,  
        nethack.Command.SEARCH,
        nethack.Command.PICKUP 
    )

    pixel_crop_size = pixel_crop_rows*pixel_crop_cols

    hyper_params_dqn = {
        "seed": 42,  # which seed to use
        "env": "MiniHack-Quest-Hard-v0",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-3,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 256,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "num-runs": 20 # used to average reward per episode over 20 runs
    }

    np.random.seed(hyper_params_dqn["seed"])
    random.seed(hyper_params_dqn["seed"])

    # initialize gym environment
    env = gym.make("MiniHack-Quest-Hard-v0", observation_keys=("glyphs_crop", "pixel_crop"), actions=NAVIGATE_ACTIONS, reward_win=1, reward_lose=-1)
    env.seed(hyper_params_dqn["seed"])

    # TODO Pick Gym wrappers to use
    # get only pixel crop from environment as rgb values
    env = ColorPixelRenderingDQN(env, "pixel_crop")
    # env = gym.wrappers.Monitor(env, "videos\dqn", force=True)

    # Initialize replay memory D to capacity N
    N = hyper_params_dqn["replay-buffer-size"]
    D = ReplayBuffer(N)

    num_actions = env.action_space.n
    
    # TODO Create dqn agent
    agent = DQNAgent(pixel_crop_size, num_actions, D, use_double_dqn=hyper_params_dqn["use-double-dqn"], lr=hyper_params_dqn['learning-rate'], batch_size=hyper_params_dqn['batch-size'], gamma=hyper_params_dqn['discount-factor'], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    eps_timesteps = hyper_params_dqn["eps-fraction"] * float(hyper_params_dqn["num-steps"])
    # ave_episode_rewards = [0.0]


    from collections import defaultdict
    average_rewards = defaultdict(lambda: [])

    state = env.reset()['glyphs_crop']
    state = state.flatten()
    num_steps = hyper_params_dqn["num-steps"]
    average_reward_per_eps = 0
    steps_per_eps = 0
    num_runs = hyper_params_dqn["num-runs"]

    print("DQN:\nRuning", num_steps, "steps averaged over", num_runs, "runs...")
    
    for run in range(num_runs):
        episode_num = 0
        print("--------------Start of run", run, "--------------")
        for t in range(num_steps):
            steps_per_eps +=1
            fraction = min(1.0, float(t) / eps_timesteps)
            eps_start = hyper_params_dqn["eps-start"]
            eps_end = hyper_params_dqn["eps-end"]
            eps_threshold = eps_start + fraction * (eps_end - eps_start)
            sample = random.random()
            # TODO
            #  select random action if sample is less equal than eps_threshold
            # take step in env
            # add state, action, reward, next_state, float(done) to reply memory - cast done to float
            # add reward to episode_reward
            if(sample > eps_threshold):
                a = agent.act(state)
            else:
                a = env.action_space.sample()

            new_state, reward, done, info = env.step(a)
            new_state = new_state['glyphs_crop'].flatten()
            agent.memory.add(state, a, reward, new_state, float(done))
            state = new_state

            average_reward_per_eps += reward
            if done:
                state = env.reset()['glyphs_crop']
                state = state.flatten()
                average_reward_per_eps = average_reward_per_eps / steps_per_eps
                print("Run number", run, "       Episode number", episode_num, "       Average reward =", average_reward_per_eps)
                average_rewards[episode_num].append(average_reward_per_eps)

                # print("Episode number", num_episodes, ",     Average episode reward")

                # initialize reward for new episode
                episode_num += 1
                average_reward_per_eps = 0
                steps_per_eps = 0


            if (
                t > hyper_params_dqn["learning-starts"]
                and t % hyper_params_dqn["learning-freq"] == 0
            ):
                agent.optimise_td_loss()

            if (
                t > hyper_params_dqn["learning-starts"]
                and t % hyper_params_dqn["target-update-freq"] == 0
            ):
                agent.update_target_network()
        print("--------------End of run", run, "--------------\n")

    average_rewards_list = averageRewards(average_rewards)
    print("\nReward per episode averaged over", num_runs, "runs\n", average_rewards_list,"\n")

    #send data to text fie
    file1 = open("dqn_data.txt","a")
    file1.writelines("DQN average rewards per episode:\n")
    for i in range(len(average_rewards_list)):
        file1.writelines(str(average_rewards_list[i]))
        if i != len(average_rewards_list)-1:
            file1.writelines(",")
    file1.writelines("\n")
    file1.close() #to change file access modes