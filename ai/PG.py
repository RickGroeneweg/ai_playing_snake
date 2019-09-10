import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

from torch.distributions.categorical import Categorical

from collections import deque

from Abstract_Agent import Agent



# xavier uniform weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


# ugely copy-paste from DQN.py
class Q_Net(nn.Module):
    """neural network used by the DQN agent"""

    def __init__(self, outputs):
        super(Q_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 60, kernel_size=3)
        self.conv2 = nn.Conv2d(60, 50, kernel_size=3)
        self.conv3 = nn.Conv2d(50, 20, kernel_size=3)
        
        
        linear_input_size = 2*2*20
        self.l1 = nn.Linear(linear_input_size, 60)
        self.l2 = nn.Linear(60, 30)
        
        self.softmax = nn.Softmax()
        
        # for each action, the head gives a probability of the agent taking the action
        self.head = nn.Linear(30, outputs)
        
    def forward(self, x: torch.Tensor):   
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x_vector = x.view(x.size(0), -1)
        
      
        x = F.relu(self.l1(x_vector))
        x = F.relu(self.l2(x))
        x = self.head(x)
        return self.softmax(x)
    
    
class PG_Agent(Agent):
    
    def __init__(self, policy_net, device, env, discount_factor = 0.8, learning_rate =0.002):
        super().__init__(env, device, learning_rate)
        self.policy_net = policy_net.to(self.device) #Q_Net(n_actions).to(device)
        self.policy_net.apply(init_weights)
        self.discount_factor = discount_factor
        
        # `size averages=True` lets the algorithm interpret half of the experiences as
        # good and half as bad (on average). This way the agent will keep learning
        # even if all actions that it takes are good/bad.
        self.loss = nn.CrossEntropyLoss(reduce=False, size_average=True)
        
        self.optimizer =optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
        
    def action_from_distribution(self, probabilities):
        """sample an action (int) from the distribution with given {probabilities}"""
        action_distribution = Categorical(probabilities)
        action = action_distribution.sample()
        return action.item()
    
    
    def discounted_returns(self, rewards):
        # TODO: refactor this code since it is quite unreadible
        returns = []
        running_ret = 0
        for rew in reversed(rewards):
            ret = rew + self.discount_factor * running_ret
            returns.append(ret)
            running_ret = ret 
        return torch.tensor(list(reversed(returns))).to(self.device)
    
    def optimize(self, probabilities_batch, action_batch, rewards):
        """
        The policy gradient algorithm works by minimizing the loss function of the
        probabilities batch and the action batch. However, this loss is also 
        multiplied by the discounted reward. This encourages the probabilities
        to get closer to the taken actions if the reward is good, but further away if the
        reward is bad. 
        """
        
        discounted_returns = self.discounted_returns(rewards)
        
        #print(f'p: {probabilities_batch}')
        #print(f'a: {action_batch}')
        loss = self.loss(probabilities_batch, action_batch)
        loss = discounted_returns * loss
        
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()
        
    
    def train_episode(self, render = False, punish_end_of_game = True):
        
        # initialize environment, metaparameters and accumulators
        state = self.env.reset()
        state = torch.from_numpy(state).to(self.device, dtype=torch.float)
        done = False
        t=0
        rewards = []
        action_batch = []
        probabilities_batch = []
        
        # play a game untill the end
        while not done:
            
            if render:
                self.env.render()
            # select an action using probabilites from the policy net
            probabilities = self.policy_net(state)
            probabilities = probabilities

            
            action = self.action_from_distribution(probabilities.unsqueeze(0)) 
            
            next_state, reward, done = self.step(action)
            if punish_end_of_game:
                reward = reward if not done else -10
            
            rewards.append(reward)
            
            action_batch.append(action)
            probabilities_batch.append(probabilities)
            
            state = next_state
            t = t+1
        
        action_batch = torch.tensor(action_batch).to(self.device)

        probabilities_batch = torch.stack(probabilities_batch) 
  
        self.optimize(probabilities_batch, action_batch, rewards)

        if render:
            print(f'total reward: {sum(rewards)}')
            
            
        
        
        
    
    
