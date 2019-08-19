"""
Reinforcement learning agent for Snake
"""
from collections import namedtuple
import random
import math
from itertools import count



import gym
import gym_snake


# pyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('snake-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")




# type alias
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    memory for the agent, stores trasitions to train the NN with.
    basically the same code as the dqn example of the pyTorch docs
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
    
    
class DQN(nn.Module):
    """neural network used by the DQN agent"""

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1)
        
        linear_input_size = 4*4*10
        self.l1 = nn.Linear(linear_input_size, 80)
        self.l2 = nn.Linear(80, 40)
        
        self.head = nn.Linear(40, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        x = F.relu(self.l2(x))
        return self.head(x)
    
steps_done = 0


BATCH_SIZE = 1000
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.0
EPS_DECAY = 100
TARGET_UPDATE = 10  

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(10, 10, n_actions).to(device)
target_net = DQN(10, 10, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.SGD(policy_net.parameters(), lr=0.02)
memory = ReplayMemory(10000)


def select_action(state):
  
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        #exploitation
        
        
        with torch.no_grad():
            #state = torch.from_numpy(env.state()).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        #exploration
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []    
  

def optimize_model():
    if len(memory) < BATCH_SIZE :
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).to(device, dtype=torch.float)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()    
    
    
def main():
    num_episodes = 1_000_000
    #scores = []
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
    
        state = env.state()
        state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
        for t in count():
            # Select and perform an action
            action = select_action(state)

            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.from_numpy(next_state).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
            if i_episode % 6 == 0:
                env.render()

            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.close()
    
if __name__ == '__main__':
    main()
    
