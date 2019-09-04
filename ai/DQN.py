import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        return self.head(x)
    
    
class DQN_Agent:
    def __init__(self, n_actions, device, gamma, lr):
        self.device = device
        self.target_net = Q_Net(n_actions).to(self.device)
        self.policy_net = Q_Net(n_actions).to(self.device)
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=lr)
        self.n_actions = n_actions
        self.gamma = gamma
        self.actions = deque(maxlen=200)
        
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

    def optimize(self, batch):
        self.optimizer.zero_grad()
        
        loss = self.compute_loss(batch)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()       
        
    def select_action(self, state, eps_threshold):
        """      
        if the sample is below the threshold, we do exploration. With threshhold of 0 there is no exploration
        """
        
        sample = random.random()
        
        if sample < eps_threshold:
            return self._select_action_exploration()
        
        # else 
        action = self._select_action_exploitation(state)
        self.actions.append(action)
        return action

    def _select_action_exploitation(self, state):
        
        state = torch.from_numpy(state).to(self.device, dtype = torch.float).unsqueeze(0).unsqueeze(0)
        
        assert state.shape == (1, 1, 8,8), f"state.shape: {state.shape}"
        
        with torch.no_grad():
            # t.max(1) will return 
            #   [0]: largest column value of each row,and 
            #   [1]: the index where this value was found
            # so we pick action with the larger expected reward. 
            #print(self.policy_net(state))
            policy_output =self.policy_net(state)
            action = policy_output.max(1)[1].view(1, 1)
            #print(action.item())
            #print(f'selecting action, net: {net(state)}')
            return action.item()

    def _select_action_exploration(self):
        """just return a random action from the action space"""
        return random.randrange(self.n_actions) 
    
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