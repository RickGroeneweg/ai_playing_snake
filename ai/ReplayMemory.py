from collections import namedtuple
import random

import torch
from torch.utils import data

# type alias
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(data.Dataset):
    """
    memory for the agent, stores trasitions to train the NN with.
    basically the same code as the dqn example of the pyTorch docs, augmented with
    some custom sampling tricks
    
    API:
        `push` to save transitions
        `sample` to sample a batch of transitions
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.rewards = [] # I seperated different kinds of events to create a more balanced replay learning
        self.punishments = []
        self.neutral = []
        
    @staticmethod    
    def rotate_transition(transition, k=1):
        """k is how many times the transition is rotated by 90 degrees counter clockwise"""
        rot_state_world = torch.rot90(transition.state[1].squeeze(0).squeeze(0), k).unsqueeze(0).unsqueeze(0)
        rot_state = (transition.state[0], rot_state_world)
        
        rot_action = (transition.action - k) % 4 # +1 is a clockwise rotation
        if transition.next_state is not None:
            rot_next_state_world = torch.rot90(transition.next_state[1].squeeze(0).squeeze(0), k).unsqueeze(0).unsqueeze(0)
            rot_next_state = (transition.next_state[0], rot_next_state_world)
        else:
            rot_next_state = None
            
        result = Transition(rot_state, rot_action, rot_next_state, transition.reward)
        
        return result
        
        
    def all_memories(self):
        return self.rewards + self.punishments + self.neutral
        
    def longest_memory(self):
        """return the memory list which is longest"""
        memory_lists = [self.rewards, self.punishments, self.neutral]
        max_len = 0
        at_idx = 0
        for idx, ls in enumerate(memory_lists):
            if len(ls)>= max_len:
                max_len = len(ls)
                at_idx = idx
        return memory_lists[at_idx]
          
        
    def push(self, *args):
        """Saves a transition."""
        trans = Transition(*args)
        
        if trans.reward == 1:
            self.rewards.append(trans)
        elif trans.reward < 0:
            self.punishments.append(trans)
        else:
            self.neutral.append(trans)
            
        if len(self) > self.capacity:
            # we remove the oldest memory from the longest buffer, to ceep the memory balanced
            self.longest_memory().pop(0)
            assert len(self) == self.capacity

    def sample(self, batch_size, rotating_k: int = 0):
        result = random.sample(self.all_memories(), batch_size)
        #if rotating_k > 0:
        result = [self.rotate_transition(transition, k=rotating_k) for transition in result]
        return result
        
    
    def __getitem__(self, index):
        return (self.rewards + self.punishments + self.neutral)[index]

    def __len__(self):
        return len(self.rewards) + len(self.punishments) + len(self.neutral)