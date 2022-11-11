import gym
import minihack
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import random

class Reinforce:
    def __init__(self, env, hyperparameters):
        #set env
        self.env = env
        observation = self.env.reset()

        #set hyperparameters
        self.learning_rate = hyperparameters["learning_rate"]
        self.num_episodes = hyperparameters["num_episodes"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.PATH = hyperparameters["PATH"] #path where the policy will be saved after being trained
                                            #path can also be used to load a policy saved in that path

        #set state size
        self.state_size = len(self.get_state(observation))

        #set action size
        self.num_actions = 5
        
        #set action array
        self.action_array = [0,1,2,3,4]# 4 = Kick

        #set policy using state size and number of actions
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.num_actions),
            torch.nn.Softmax(dim=-1)
        )

        #set device used for pytorch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        #move model to device
        self.policy.to(self.device)

        #set policy optimizer
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def get_state(self, observation):
        state = np.array(observation["chars_crop"]).flatten()
        return state

    def get_map(self, observation):
        map = np.array(observation["chars"]).flatten()
        return map

    #makes sure that the chosen action isnt walking into a wall
    def action_is_valid(self,action, map, player_row, player_col):
        #check edges
        if action == 0 and player_row == 0:
            return False
        if action == 1 and player_col == 78:
            return False
        if action == 2 and player_row == 20:
            return False
        if action == 3 and player_col == 0:
            return False


        if action == 0:
            if map[(player_row-1)*79 + player_col] == 32:
                return False
        if action == 1:
            if map[(player_row)*79 + player_col+1] == 32:
                return False
        if action == 2:
            if map[(player_row+1)*79 + player_col] == 32:
                return False
        if action == 3:
            if map[(player_row)*79 + player_col-1] == 32:
                return False
        return True

    def update_player_position(self,player_row, player_col, action):
        if action == 0:
            return (player_row-1, player_col)
        if action == 1:
            return (player_row, player_col+1)
        if action == 2:
            return (player_row+1, player_col)
        if action == 3:
            return (player_row, player_col-1)
        return (player_row, player_col)

    def get_random_action(self):
        return random.randrange(self.num_actions)

    def train(self):
        e = 0.3
        losses = []
        average_reward_per_episode = []
        for i in range(self.num_episodes):
            print("training episode: ", i)

            rewards = []

            memory = []
            observation = self.env.reset()

            map = self.get_map(observation)

            #find player position
            for j in range(21):
                for k in range(79):
                    if map[j*79 + k] == 64:
                        player_row = j
                        player_column = k
                
            
            done = False
            while not done:
                state = torch.tensor(self.get_state(observation), dtype=torch.float)
                map = self.get_map(observation)

                action_probabilities = self.policy(state.to(self.device)) #get action prob. by passing observation to the policy neural network
                action_distribution = torch.distributions.Categorical(probs=action_probabilities) #create distr. of actions
                action = action_distribution.sample().item() #choose action from distr.

                #the below section contains the code for checking valid actions
                '''
                #check if action is illegal (walking into a wall)
                #this step is to encourage exploration and prevent local minima being found
                valid_action = False
                while not valid_action:
                    if self.action_is_valid(action, map, player_row, player_column):
                        valid_action = True
                    else:
                        action = self.get_random_action()
                player_row, player_column = self.update_player_position(player_row, player_column, action)
                '''

                #the below section contains the code for choosing a random action based on epsilon
                '''
                #choose random action with probability e
                if random.randrange(100)<e*100:
                    action = self.get_random_action()
                '''

                #get action from action array
                #this will be passed to the environment
                nethack_action = self.action_array[action]
                
                #take action and record step
                observation_, reward, done, _ = self.env.step(nethack_action)

                rewards.append(reward)

                #the below section contains the code for negatively rewarding unchanged state
                '''
                #negatively reward action that results in no state change
                if np.array_equal(self.get_state(observation), self.get_state(observation_)):
                    reward = reward - 0.05
                '''
                
                memory.append((state, torch.tensor(action, dtype=torch.int), reward))

                observation = observation_
            #appending the last move to memory
            memory.append((state, torch.tensor(action, dtype=torch.int), reward))

            sum = 0
            for j in range(len(rewards)):
                sum = sum + rewards[j]
            average_reward_per_episode.append(sum/len(rewards))
        
            G = 0
            last = True
            for state, action, reward in reversed(memory):
                if last:
                    #not going to calculate discounted return for the end state
                    last = False
                else:
                    #calculate reward + discounted future returns for each state (discounted return)
                    G = self.discount_factor*G + reward
                    
                    action_probabilities = self.policy(state.to(self.device)) #get action prob. by passing observation to the policy neural network
                    action_distribution = torch.distributions.Categorical(probs=action_probabilities) #create distr. of actions

                    #get the log probability of choosing sample action given the policy
                    log_prob = action_distribution.log_prob(action.to(self.device))

                    #log_prob is made negative because we want to maximise the log likelihood.
                    #gradient decent attempts to minimize (move down the gradient), therefore minimizing the negative of the log_prob
                    #will maximise the log_prob term
                    loss = 0-log_prob*G

                    losses.append(loss.item())

                    self.policy_optimizer.zero_grad()
                    loss.backward()
                    self.policy_optimizer.step()
        #save policy            
        torch.save(self.policy.state_dict(), self.PATH)

        #plot loss over training
        plt.plot(losses)
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.title("Learning rate %f"%(self.learning_rate))
        plt.savefig('loss.png')    

        #plot average reward
        plt.plot(average_reward_per_episode)
        plt.ylabel('average reward')
        plt.xlabel('episode')
        plt.title("Learning rate %f"%(self.learning_rate))
        plt.savefig('training_average_reward.png')    
    
    def load_policy(self):
        self.policy.load_state_dict(torch.load(self.PATH))

    def test_policy(self):   
        #play through 100 episodes and average reward   
        average_reward_per_episode = []  
        for i in range(100):
            print("test episode: ", i)
            rewards = []
            observation = self.env.reset()
            
            '''
            #render env
            self.env.render()
            '''
            step = 0
            max_steps = 5000
            done = False
            while not done and step < max_steps:
                state = torch.tensor(self.get_state(observation), dtype=torch.float)

                action_probabilities = self.policy(state.to(self.device)) #get action prob. by passing observation to the policy neural network
                action_distribution = torch.distributions.Categorical(probs=action_probabilities) #create distr. of actions
                action = action_distribution.sample().item() #choose action from distr.

                #get action from action array
                #this will be passed to the environment
                nethack_action = self.action_array[action]

                #take action
                observation_, reward, done, _ = self.env.step(nethack_action)
                rewards.append(reward)
                '''
                #render env
                time.sleep(1)
                self.env.render()
                '''
                observation = observation_
                step+=1
            sum = 0
            for j in range(len(rewards)):
                sum = sum + rewards[j]
            average_reward_per_episode.append(sum/len(rewards))
        
        #plot average reward
        plt.plot(average_reward_per_episode)
        plt.ylabel('average reward')
        plt.xlabel('episode')
        plt.title("Learning rate %f"%(self.learning_rate))
        plt.savefig('test_average_reward.png')   
                    