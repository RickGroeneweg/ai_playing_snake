import snake_backend
import gym
import numpy as np
from gym import spaces

import tkinter as tk

class SnakeEnv(gym.Env):
    """OpenAI gym environment for Snake, using the snake_backend package"""
    
    action_decoder = {0: (0,-1), 1: (1,0), 2: (0,1), 3: (-1,0)}

    def __init__(self):
        super(SnakeEnv, self).__init__()
        
        # Initialize Tk for rendering
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=60, height=60, bg='blue')
        self.canvas.pack()
        
        # Initialize game
        self.game = snake_backend.GameOfSnake([(0,0), (0,1), (0,2)], hasFrontEnd=False)

        self.action_space = spaces.Discrete(4)


    def step(self, action: int):
        direction = self.action_decoder[action]
        obs, rew, done = self.game.step(direction)
        
        return obs, rew, done, ""
    
    def state(self):
        return self.game.array()
        
    def reset(self):
        self.game = snake_backend.GameOfSnake([(0,0), (0,1), (0,2)], hasFrontEnd=False)


    def render(self, mode='human', close=False):
        self.canvas.delete('all')
        height, width = self.state().shape

        for y in range(height):
            for x in range(width):
                num = self.state()[y, x]
                if num > 0:
                    self.canvas.create_oval(x*10, y*10, x*10+10, y*10+10, fill = 'red')
                elif num == -1:
                    self.canvas.create_oval(x*10, y*10, x*10+10, y*10+10, fill = 'green')
        self.root.update()
        
        
