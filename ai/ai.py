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
        self.conv1 = nn.Conv2d(3, 5, kernel_size=3)
        
        
        
        linear_input_size = 3*3*5
        self.l1 = nn.Linear(linear_input_size, 20)
        self.l2 = nn.Linear(20, 10)
 
        self.head = nn.Linear(10, outputs)
        
        self.softmax = nn.Softmax()
        
        
        
    def forward(self, x: torch.Tensor):   
        x = x
        x = F.leaky_relu(self.conv1(x))
        x_vector = x.view(x.size(0), -1)
        
      
        x = F.relu(self.l1(x_vector))
        x = F.relu(self.l2(x))
        x = self.head(x)
        return self.softmax(x)
    
    def forward_1(self, x):
        "forward a sample of only one"
        x = x.unsqueeze(0)
        x = self.forward(x)
        return x.squeeze(0)

policy_net = Q_Net(4)
agent = PG_Agent(policy_net, device, env, discount_factor = 0.6, learning_rate =0.001)


def main(N_episodes = 5_000_000):        
    #agent.update_learning_rate(learning_rate)
 
    # cycle through playing a game and optimizing
    eps = 0.2
    for i in range(N_episodes):
        # make the threshold go down over time, but fluctuate
        eps = abs((1 - i/N_episodes) * sin(0.001*i))*0.5
        
        render: bool = (i % 400 ==0 and eps <0.1)

        # play one game
        agent.train_episode(render=render, eps=eps)
        # train the parameters of the agent's policy net
        #agent.optimize()       
        
        agent.end_episode(i, summary = render)


    print('Complete')
    env.close()
    
if __name__ == '__main__':
    main()
    
