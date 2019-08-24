"""
Reinforcement learning script for Snake
"""

import random
import math
from itertools import count



import gym
import gym_snake #s omehow we do need this import


# pyTorch imports
import torch
import torch.optim as optim
import torch.nn.functional as F

from ReplayMemory import ReplayMemory
from DQN import DQN

env = gym.make('snake-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

    

    
steps_done = 0 # ugly global counter

# meta parameters
BATCH_SIZE = 100
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.005
EPS_DECAY = 2000
TARGET_UPDATE = 10  

# Get number of actions from gym action space
n_actions = env.action_space.n

# here we define the neural nets
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
       return _select_action_exploitation(state)
    else:
       return _select_action_exploration()
        


def _select_action_exploitation(state):
    """run the state through the policy network to get an action"""
    
    hunger, tensor = state
    hunger = torch.Tensor([[hunger]]).to(device, dtype=torch.float)
    
    with torch.no_grad():
        # t.max(1) will return 
        #   [0]: largest column value of each row,and 
        #   [1]: the index where this value was found
        # so we pick action with the larger expected reward.
        return policy_net(tensor, hunger).max(1)[1].view(1, 1)

def _select_action_exploration():
    """just return a random action from the action space"""
    return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    


episode_durations = []    
  

def optimize_model(memory, optimizer, batch_size= BATCH_SIZE):
    if len(memory) < batch_size:
        print('not yet enough memory for replay')
        return
    
    rotating_k = random.randint(0, 3)
    transitions = memory.sample(BATCH_SIZE, rotating_k)
    loss_acc = 0
    
    for transition in transitions:
        hunger, world = transition.state
    
        
        Q_value = policy_net(world, hunger).gather(1, transition.action)
        Q_bellman = estimated_Q(transition.reward, transition.next_state)
        

        
        
        loss = F.smooth_l1_loss(Q_value, Q_bellman)
        loss_acc += loss
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    
    print(f'average loss: {loss_acc/len(transitions)}')
    
def estimated_Q(reward, next_state: tuple):
    reward = reward.to(dtype=torch.float).unsqueeze(0)
    if next_state is None:
        # the game has ended, so the Q value is just the reward
        return reward
    
    else:
        with torch.no_grad():  
            # we are not training the target_net, we are updating it with paramters form the policy_net, 
            # so no differentiation is needed here.
            next_expected_Q = target_net(next_state[1], next_state[0]).max(1)[0]
            return reward + GAMMA*next_expected_Q


def take_step_(action, rendering=False):
    next_state, reward, done, _ = env.step(action.item())
    if next_state is not None:
        next_state_tensor = torch.from_numpy(next_state[1]).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
        hunger = torch.Tensor([[next_state[0]]]).to(device, dtype=torch.float)
        next_state = hunger, next_state_tensor
    if rendering:
        env.render()
        
    reward = torch.tensor([reward], device=device)
    return next_state, reward, done
    
def main(num_episodes = 10_000_000):
    
 
    #scores = []
    for i_episode in range(num_episodes):
        rendering = i_episode % 50 == 0
        # Initialize the environment and state
        env.reset()
        if rendering:
            env.render()
    
        state = env.state()
        hunger, tens = state
        tens = torch.from_numpy(tens).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
        hunger = torch.Tensor([[hunger]]).to(device, dtype=torch.float) # ugly copy paste
        state = (hunger, tens)
        
        for t in count():
            # Select and perform an action
            action = select_action(state)
           

            # let the snake take a step by performing {action}
            next_state, reward, done = take_step_(action, rendering = rendering)
           

            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(memory, optimizer)
         
            
            if done:
                # game over
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.close()
    
if __name__ == '__main__':
    main()
    
