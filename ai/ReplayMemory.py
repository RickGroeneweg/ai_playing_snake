from collections import namedtuple
import random

import torch
from torch.utils import data

import numpy as np

# type alias
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(data.Dataset):
    """
    memory for the agent, stores trasitions to train the NN with.
    basically the same code as the dqn example of the pyTorch docs, augmented with
    some custom sampling tricks
    
    API:
        `push` to save transitions
        `sample` to sample a batch of transitions
    """

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.memory = []
        self.device = device
        
    def print_action_distribution(self):
        actions = [t.action for t in self.memory]
        ls = [0,0,0,0]
        for a in actions:
            ls[a.item()] += 1
        print(f'action distribution: {ls}')
            
        
    @staticmethod    
    def rotate_transition(transition, k=1):
        """k is how many times the transition is rotated by 90 degrees counter clockwise"""
        rotated_state = torch.rot90(transition.state.squeeze(0).squeeze(0), k).unsqueeze(0).unsqueeze(0)
        
        rotated_action = (transition.action - k) % 4 # +1 is a clockwise rotation
        
        rotated_next_state =  torch.rot90(transition.next_state.squeeze(0).squeeze(0), k).unsqueeze(0).unsqueeze(0)
            
        result = Transition(rotated_state, rotated_action, rotated_next_state, transition.reward, transition.done)
        
        return result

        
    def push(self, state, action, next_state, reward, done):
        """Saves a transition."""
        assert isinstance(state, np.ndarray), f'type: {type(state)}'
        assert isinstance(action, int), f'type: {type(action)}'
        assert isinstance(next_state, np.ndarray), f'type: {type(next_state)}'
        assert isinstance(done, bool), f'type: {type(done)}'
               
        state = torch.from_numpy(state).to(device=self.device, dtype=torch.float)
        next_state = torch.from_numpy(next_state).to(device=self.device, dtype=torch.float) 
        reward = torch.Tensor([reward]).to(self.device)
        done = torch.Tensor([done]).to(self.device)
        action = torch.Tensor([action]).to(self.device, dtype = torch.long)
        
        trans = Transition(state, action, next_state, reward, done)
        
        self.memory.append(trans)
            
        if len(self) > self.capacity:
            # we remove the oldest memory from the longest buffer, to ceep the memory balanced
            self.memory.pop(0)
            assert len(self) == self.capacity

    def sample(self, batch_size, k: int = 0):
        """
        get a list of {batch_size} transitions from the replay memory, randomly selected.
        The transitions will be rotated by 90 degrees {k} times because the game is rotation
        symmetric and this leads to more robust learning.
        """
        batch_size = min(batch_size, len(self.memory))
        
        result_ = random.sample(self.memory, batch_size)
        #if rotating_k > 0:
        result = [self.rotate_transition(transition, k=k) for transition in result_]
        
        # TODO: there is a bug here in the shape of the tensors!!!!
        return result
        
    
    def __getitem__(self, index):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)