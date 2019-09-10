import random
import torch

class Agent:
    def __init__(self, env, device, learning_rate):

        self.env = env
        self.device = device
        self.learning_rate = learning_rate
        
    def create_Q_net(self):
        """
        creates a neural net that takes the state of self.env and outputs a value
        for each possible action
        """
        pass
        
    def update_learning_rate(self, learning_rate):
        """set all learning-rate parameters of the optimizer to the given learning_rate"""
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate  
            
    def end_episode(self, i, summary=False):
        """
        executes at the end of an episode, some algorithms have some unfinished 
        things or do some updates here.        
        """
        if summary:
            print(f'***episode number: {i}***')
            #print(f'ε_threshold: {ε_threshold}')
            #self.print_avg_loss()
            #self.print_action_distribution()     
        
    def optimize(self):
        raise NotImplementedError()
        
    def train_episode(self):
        raise NotImplementedError()
        
    def step(self, action, render=False):    
        """take a step in the environment. N.B. this changes the state of self.env"""
        next_state, reward, done, _ = self.env.step(action)
        next_state = torch.from_numpy(next_state).to(self.device, dtype=torch.float)
    
        if render:
            self.env.render()
    
        return next_state, reward, done
   
    def select_action_epsilon_greedily(self, state, eps_threshold):
        """use an epsilon-greedy policy to select an action for the given state"""
        sample = random.random() # sample from standard uniform distribution 
        if sample < eps_threshold:
            return self._select_action_exploration()
        else:
            return self._select_action_exploitation(state)
        

    def _select_action_exploitation(self, state):
        """
        use the policy network to give values to each action, and pick
        the action with the highest value
        """
        
        with torch.no_grad():
            policy_output =self.policy_net(state)
            # t.max(1) will return 
            #   [0]: largest column value of each row,and 
            #   [1]: the index where this value was found
            # so we pick action with the larger expected reward. 
            action = policy_output.max(1)[1].view(1, 1)
            return action.item()

    def _select_action_exploration(self):
        """just return a random action from the action space"""
        return random.randrange(self.env.action_space.n) 