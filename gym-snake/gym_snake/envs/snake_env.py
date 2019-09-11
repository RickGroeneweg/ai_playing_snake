import snake_backend
import gym
from gym import spaces
import random
import tkinter as tk

class SnakeEnv(gym.Env):
    """OpenAI gym environment for Snake, using the snake_backend package"""
    

    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.reset()
        # Initialize Tk for rendering
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.game.X*30, height=self.game.Y*30, bg='blue')
        self.canvas.pack()
        
        

    def step(self, action: int):
        direction = snake_backend.Direction(action)
        state, reward, done = self.game.step(direction)
        self.score += reward
        
        return state, reward, done, ""
    
    def state(self):
        """returns a tuple of hunger: int and screen: np.array"""
        return self.game.array()
        
    def reset(self): 
        self.game = snake_backend.GameOfSnake(hasFrontEnd=False)
        self.action_space = spaces.Discrete(4)
        self.score = 0
        
        return self.game.array()

    def set_state(self, array):
        assert array.shape == (self.game.X, self.game.Y), f'{array.shape} not equal to {(self.game.X, self.game.Y)}'
        pass
        


    def render(self, mode='human', close=False):
        self.canvas.delete('all')
        _, height, width = self.state().shape
        


        for y in range(height):
            for x in range(width):
                snake_num = self.state()[0, y, x]
                food_num = self.state()[2,y,x]

                if snake_num == 1:
                    self.canvas.create_oval(x*30, y*30, x*30+30, y*30+30, fill = 'red')
                if food_num == 1:
                    self.canvas.create_oval(x*30, y*30, x*30+30, y*30+30, fill = 'green')
                    
        self.canvas.create_text(100,200,fill="white",font="Times 20 italic bold",
                        text=f'score: {self.score}')

        self.root.update()
        
        
