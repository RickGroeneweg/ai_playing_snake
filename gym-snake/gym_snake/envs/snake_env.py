import snake_backend
import gym
from gym import spaces
import random
import tkinter as tk

class SnakeEnv(gym.Env):
    """OpenAI gym environment for Snake, using the snake_backend package"""
    

    def __init__(self):
        super(SnakeEnv, self).__init__()
        
        # Initialize Tk for rendering
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=240, height=240, bg='blue')
        self.canvas.pack()
        
        self.reset()

    def step(self, action: int):
        direction = snake_backend.Direction(action)
        obs, rew, done = self.game.step(direction)
        self.score += rew
        
        return obs, rew, done, ""
    
    def state(self):
        """returns a tuple of hunger: int and screen: np.array"""
        return self.game.to_state()
        
    def reset(self): 
        self.game = snake_backend.GameOfSnake([(3,0), (3,1), (3,2)], food = (3+random.randint(0, 1), 3+random.randint(0, 1)), hasFrontEnd=False)
        self.action_space = spaces.Discrete(4)
        self.score = 0


    def render(self, mode='human', close=False):
        self.canvas.delete('all')
        height, width = self.state()[1].shape


        for y in range(height):
            for x in range(width):
                num = self.state()[1][y, x]

                if num == 0.2:
                    self.canvas.create_oval(x*30, y*30, x*30+30, y*30+30, fill = 'green')
                elif num > 0:
                    self.canvas.create_oval(x*30, y*30, x*30+30, y*30+30, fill = 'red')
                elif num == -1:
                    self.canvas.create_rectangle(x*30, y*30, x*30+30, y*30+30, fill = 'black')
                    
        hunger = self.state()[0]
        self.canvas.create_rectangle(10, 10, 10 + hunger, 20, fill='yellow')
        self.canvas.create_text(100,200,fill="white",font="Times 20 italic bold",
                        text=f'score: {self.score}')

        self.root.update()
        
        
