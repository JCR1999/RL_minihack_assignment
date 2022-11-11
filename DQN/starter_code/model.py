from gym import spaces
import torch.nn as nn


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, input_size, n_actions):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        # assert (
        #     type(observation_space) == spaces.Box
        # ), "observation_space must be of type Box"
        # assert (
        #     len(observation_space.shape) == 3
        # ), "observation space must have the form channels x width x height"
        # assert (
        #     type(action_space) == spaces.Discrete
        # ), "action_space must be of type Discrete"

        # TODO Implement DQN Network

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_size , out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256 , out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256 , out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_actions)
        )

    def forward(self, x):
        # TODO Implement forward pass
        return self.fc(x)
