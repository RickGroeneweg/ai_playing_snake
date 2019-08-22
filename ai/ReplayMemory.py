from collections import namedtuple
import random

# type alias
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
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
        self.position = 0
        
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
        
        if trans.reward == 100:
            self.rewards.append(trans)
        elif trans.reward < 0:
            self.punishments.append(trans)
        else:
            self.neutral.append(trans)
            
        if len(self) > self.capacity:
            self.longest_memory().pop(0)
            assert len(self) == self.capacity

    def sample(self, batch_size):
        return random.sample(self.all_memories(), batch_size)

    def __len__(self):
        return len(self.rewards) + len(self.punishments) + len(self.neutral)