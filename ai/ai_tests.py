import unittest
import gym
import gym_snake # somehow we do need this import
import torch

from ai import _select_action_exploration

class TestMainScript(unittest.TestCase):
    
    def test_state_type(self):
        """we encode the state of the game by a tuple of the hunger and the world"""

        env = gym.make('snake-v0')
        env.reset()
        state = env.state()
        
        self.assertIsInstance(state, tuple)
        
    def test_select_action_exploration(self):
        action = _select_action_exploration()
        self.assertIsInstance(action, torch.Tensor)
        self.assertIsInstance(action.item(), int)