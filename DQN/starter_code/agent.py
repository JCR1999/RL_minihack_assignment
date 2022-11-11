from gym import spaces
import numpy as np

from starter_code.model import DQN
from starter_code.replay_buffer import ReplayBuffer
import torch.nn.functional as Fun_torch_nn
import torch

device = "cuda"


class DQNAgent:
    def __init__(
        # self,
        # observation_space: spaces.Box,
        # action_space: spaces.Discrete,
        # replay_buffer: ReplayBuffer,
        # use_double_dqn,
        # lr,
        # batch_size,
        # gamma,

        self,
        pixel_crop_size,
        n_actions,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
        device=torch.device("cuda")
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        # TODO: Initialise agent's networks, optimiser and replay buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory = replay_buffer
        self.use_double_dqn = use_double_dqn
        self.target_network = DQN(pixel_crop_size, n_actions).to(device)
        self.policy_network = DQN(pixel_crop_size, n_actions).to(device)
        self.update_target_network()
        self.target_network.eval()
        self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=lr)        
        self.device = device

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # TODO
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss

        device = self.device
        s, a, r, ns, dones = self.memory.sample(self.batch_size)
        s = np.array(s)
        ns = np.array(ns)
        s = torch.from_numpy(s).float().to(device)
        r = torch.from_numpy(r).float().to(device)
        a = torch.from_numpy(a).long().to(device)
        ns = torch.from_numpy(ns).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            if self.use_double_dqn:
                _, a_max = self.policy_network(ns).max(1)
                max_q = self.target_network(ns).gather(1, a_max.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.target_network(ns)
                max_q, _ = next_q_values.max(1)
            t_qvals = r + (1 - dones) * self.gamma * max_q

        i_qvals = self.policy_network(s)
        i_qvals = i_qvals.gather(1, a.unsqueeze(1)).squeeze()

        loss = Fun_torch_nn.smooth_l1_loss(i_qvals, t_qvals)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del s
        del ns
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # TODO Select action greedily from the Q-network given the state
        device = self.device
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()

    def save(self, file_name):
        torch.save(self.policy_network, file_name)
