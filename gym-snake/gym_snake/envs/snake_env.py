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
        self.canvas = tk.Canvas(self.root, width=240, height=240, bg='blue')
        self.canvas.pack()
        
        # Initialize game
        self.game = snake_backend.GameOfSnake([(0,0), (0,1), (0,2)], hasFrontEnd=False)

        self.action_space = spaces.Discrete(4)


    def step(self, action: int):
        direction = self.action_decoder[action]
        obs, rew, done = self.game.step(direction)
        
        return obs, rew, done, ""
    
    def state(self):
        """returns a tuple of hunger: int and screen: np.array"""
        return self.game.to_state()
        
    def reset(self):
        self.game = snake_backend.GameOfSnake([(0,0), (0,1), (0,2)], hasFrontEnd=False)


    def render(self, mode='human', close=False):
        self.canvas.delete('all')
        height, width = self.state()[1].shape
        print(f'height: {height}, width: {width}')

        for y in range(height):
            for x in range(width):
                num = self.state()[1][y, x]

                if num == 10:
                    self.canvas.create_oval(x*30, y*30, x*30+30, y*30+30, fill = 'green')
                elif num > 0:
                    self.canvas.create_oval(x*30, y*30, x*30+30, y*30+30, fill = 'red')
                elif num == -10:
                    self.canvas.create_rectangle(x*30, y*30, x*30+30, y*30+30, fill = 'black')
                    
        hunger = self.state()[0]
        self.canvas.create_rectangle(10, 10, 10 + hunger, 20, fill='yellow')

        self.root.update()
        
        
