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
from DQN import Q_Net, DQN_Agent

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

memory = ReplayMemory(10000, device)

agent = DQN_Agent(env.action_space.n, device, GAMMA, lr =0.01)


    


def take_step_(action, state, rendering=False):
    
    next_state, reward, done, _ = env.step(action)
    
    if rendering:
        env.render()
    
    assert state.shape == next_state.shape
    memory.push(state, action, next_state, reward, done)
    

    return next_state, reward, done
    
def main(num_episodes = 10_000_000, learning_rate = 0.005, eps_init = 1.0):

    # set learning rate of the optimizer.
    for g in agent.optimizer.param_groups:
        g['lr'] = learning_rate
    eps_threshold = eps_init    
 
    #scores = []
    for i_episode in range(num_episodes):
        eps_threshold = abs((1 - i_episode/num_episodes) * math.sin(0.001*i_episode))
        rendering = i_episode % 200 == 0
        if rendering:
            print(f'***episode number: {i_episode}***')
            print(f'epsilon threshold: {eps_threshold}')
            agent.print_avg_loss()
            
            memory.print_action_distribution()        
        
        
        # Initialize the environment and state
        reward_acc = 0
        state = env.reset()
        
        assert state.shape == (8,8)

        if rendering:
            env.render()
        
        for t in count():
            # Select and perform 
            # If we are rendering, then we do not want random exploration
            action : int = agent.select_action(state, 0 if rendering else eps_threshold)
            

            # let the snake take a step by performing {action}
            next_state, reward, done = take_step_(action, state, rendering = rendering)
           
            assert next_state.shape == (8,8)

            reward_acc += reward     
            

            # Move to the next state
            state = next_state         
            
            if done:
                # game over
                if rendering: 
                    print(f'accumulated reward: {reward_acc}')
                break
            
            if t > abs(reward_acc*100) +12:
                break
        
        
        sample = memory.sample(BATCH_SIZE, k=random.randint(0,4))


        agent.optimize(sample)
        
        
        
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print('Complete')
    env.close()
    
if __name__ == '__main__':
    main()
    
