import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from Abstract_Agent import Agent

from collections import deque

import random

class Q_Net(nn.Module):
    """neural network used by the DQN agent"""

    def __init__(self, outputs):
        #TODO: incorporate h and w so that linear_input_size is not hard coded!!!!
        super(Q_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=3)
        self.conv2 = nn.Conv2d(50, 47, kernel_size=3)
        self.conv3 = nn.Conv2d(47, 18, kernel_size=3)
        
        
        linear_input_size = 2*2*18
        self.l1 = nn.Linear(linear_input_size, 50)
        self.l2 = nn.Linear(50, 20)
        
        
        # the head gives an estimation of the Q-value for each possible action
        self.head = nn.Linear(20, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor):   
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x_vector = x.view(x.size(0), -1)
        
      
        x = F.relu(self.l1(x_vector))
        x = F.relu(self.l2(x))
        # added softmax, not tested!
        x = self.head(x)

        return x
    
    
class DQN_Agent(Agent):
    """
    A reinforcement learning agent, that operates by the DQN algorithm. Later I adjusted this agent to operate by the Double DQN algorithm.
    There seem to be multiple explanations of how the Double DQN algorithm works, that contradict each other slightly. Here I have gone for the one
    that is most simple to me:
        
        There are two neural nets, as in the vanilla DQN, but they are used slightly differently. The reason for this, is that in the vanilla 
        DQN algoritm, during optimization, the target net is used to assess how good an action is, and to select the best action (by taking the maximum). 
        This results in overestimation of how good this action actually is. In this version od the Double DQN, we use the policy net to select the best action,
        and the target net to assess how good this is.
        
        A simple way to think about this is: The policy net is used for taking decisions and selecting actions. The target net is only used to
        assess how good a certain action is (i.e. estimate Q(s,a) )
    """
    def __init__(self, n_actions, device, gamma, lr, env, memory):
        self.device = device
        self.target_net = Q_Net(n_actions).to(self.device)
        self.policy_net = Q_Net(n_actions).to(self.device)
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=lr)
        self.n_actions = n_actions
        self.gamma = gamma
        self.actions = deque(maxlen=200)
        self.env = env
        self.memory = memory
        self.target_net_update = 50
        
        # set the weights of the target net to be the same as the policy net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # to keep track of how wel the agent is doing
        self.losses = deque(maxlen=200)

    def print_avg_loss(self):
        if not self.losses:
            return
        avg_loss = sum(self.losses)/len(self.losses)
        print(f'average loss: {avg_loss}')

    def optimize(self, batch_size):
        batch = self.memory.sample(batch_size, k=random.randint(0,4))
        self.optimizer.zero_grad()
        
        loss = self.compute_loss(batch)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()       
        
    def select_action(self, state ,eps_threshold=0.2): # TODO: add functionality for stochastic policy and for epsilon-greede policy
        """      
        if the sample is below the threshold, we do exploration. With threshhold of 0 there is no exploration
        """
        state = torch.from_numpy(state).to(self.device, dtype = torch.float).unsqueeze(0).unsqueeze(0)
        
        action = self._select_action_epsilon_greedily(state, eps_threshold)
        
        self.actions.append(action)
        return action
    
    
    def compute_temporal_difference_Qs(self, batch):
        """
        Computes the current Q value by the policy network (Q_curr), and the Q value
        as computed by the iteration of Bellmann's equation, where, as in a Double-DQN
        network, the next Q value (Q_next) is extimated by the target network, but the
        action is chosen by the policy network.
        """
        #assert batch[0].state.shape == (8,8), f'first state in batch has wrong format: {batch[0].state.shape}'
        
        states = torch.cat([transition.state for transition in batch])
        next_states = torch.cat([transition.next_state for transition in batch])
        #rewards = torch.stack([transition.reward for transition in batch])
        dones = torch.stack([transition.done for transition in batch])
        actions = torch.tensor([[transition.action] for transition in batch]).to(self.device, dtype = torch.long)

        Q_curr= self.policy_net(states).gather(1, actions)
        best_next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        target_net_output = self.target_net(next_states)
        Q_next = target_net_output.gather(1, best_next_actions)
        
        # if the game ended, then there is no Q_next, so we multipy with 0
        Q_next *= (1-dones)
        
        return Q_curr, Q_next
    
    def compute_loss(self, batch):
        """
        Compute the loss, which is calculated by the difference between the current
        estimated Q, and the one by one iteration of the Bellmann equation.
        """
        assert isinstance(batch, list)
        
        Q_curr, Q_next = self.compute_temporal_difference_Qs(batch)
        rewards = torch.stack([transition.reward for transition in batch])
        
        Q_next.detach()
        
        loss = F.smooth_l1_loss(
                Q_curr,
                rewards + self.gamma * Q_next # Bellmann equation
                )        
        
        self.losses.append(loss)        
        return loss 
    
    def take_step(self, action, state, render=False):
        
    
        next_state, reward, done, _ = self.env.step(action)
    
        if render:
           self.env.render()
    
        return next_state, reward, done
       
    def update_target_net(self):
        """copy weights from the policy net to the target net"""
        policy_net_weights = self.policy_net.state_dict
        self.target_net.load_state_dict(policy_net_weights)
    
    def end_episode(self, i):
        if i % self.target_net_update == 0:
            self.update_target_net()
    
    def train_episode(self, render = False, ε_threshold=0.2):  
        """let the agent play one game"""
        state = self.env.reset()
        t=0
        reward_acc = 0
        done = False
        
        while not done:
            if render:
                self.env.render()
            # Select and perform 
            # If we are rendering, then we do not want random exploration
            action : int = self.select_action(state, 
                                               ε_threshold=0 if render else ε_threshold)
            

            # let the snake take a step by performing {action}
            next_state, reward, done = self.take_step(action, state, render = render)
            t += 1
            reward_acc += reward
            
            self.memory.push(state, action, next_state, reward, done)
            assert next_state.shape == (8,8)

                 
            
            # Move to the next state
            state = next_state         
            
            if t > abs(reward_acc*50) +12:
                # the snake has been playing awhile but has not been making much progress
                break
          

    
    
    
    
    
    
    