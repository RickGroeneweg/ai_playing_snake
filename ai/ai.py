"""
Reinforcement learning script for Snake
"""

import random
import math
from math import sin
from itertools import count



import gym
import gym_snake #s omehow we do need this import


# pyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ReplayMemory import ReplayMemory
from DQN import Q_Net, DQN_Agent
from PG import PG_Agent

env = gym.make('snake-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")


# meta parameters
BATCH_SIZE = 100
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.005
EPS_DECAY = 2000
TARGET_UPDATE = 10  

# Get number of actions from gym action space
n_actions = env.action_space.n

#memory = ReplayMemory(10000, device)
#
#agent = DQN_Agent(env.action_space.n, device, gamma=GAMMA, lr =0.01, env = env, memory=memory)

# ugely copy-paste from DQN.py
class Q_Net(nn.Module):
    """neural network used by the DQN agent"""

    def __init__(self, outputs):
        super(Q_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3)
        
        
        linear_input_size = 4*4*12
        self.l1 = nn.Linear(linear_input_size, 100)
        self.l2 = nn.Linear(100, 60)
        self.l3 = nn.Linear(60, 30)
        
        self.softmax = nn.Softmax()
        
        # for each action, the head gives a probability of the agent taking the action
        self.head = nn.Linear(30, outputs)
        
    def forward(self, x: torch.Tensor):   
        x = x.unsqueeze(0).unsqueeze(0)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x_vector = x.view(x.size(0), -1)
        
      
        x = F.relu(self.l1(x_vector))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.head(x)
        return self.softmax(x).squeeze(0).squeeze(0)

pol_net = Q_Net(4)
agent = PG_Agent(pol_net, device, env, discount_factor = 0.8, learning_rate =0.005)


def main(N_episodes = 100_000_000, learning_rate = 0.005):        
    agent.update_learning_rate(learning_rate)
 
    # cycle through playing a game and optimizing
    for i in range(N_episodes):
        # make the threshold go down over time, but fluctuate
        #ε_threshold = abs((1 - i/N_episodes) * sin(0.001*i))
        
        render: bool = i % 1000==0

        # play one game
        agent.train_episode(render=render)
        # train the parameters of the agent's policy net
        #agent.optimize()       
        
        agent.end_episode(i, summary = render)


    print('Complete')
    env.close()
    
if __name__ == '__main__':
    main()
    
