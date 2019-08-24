# -*- coding: utf-8 -*-
"""
Frontend and Backend for the game snake, this file can be played as is by executing it with python, or it can be used by a different front end or in our case of interest, an openai gym environment.
"""

import tkinter as tk
import numpy as np
import random
from collections import namedtuple


from enum import Enum
class Event(Enum):
    """things that can happen every turn""" # not yet used
    Wall = 1
    Self = 2
    Food = 3
    Starve = 4
    Nothing = 5
    
    
class Direction(Enum):
    """steps the snake can make"""
    Up = 0
    Right = 1
    Down = 2
    Left = 3
    
    def vec(self):
        """return unit vector where the first axis is down and the second axis is right"""
        vector_dict = {0: np.array([-1,0]),  1: np.array([0,1]), 3: np.array([0,-1]), 2: np.array([1, 0])}
        return vector_dict[self.value]
      
        
class FrontEnd:
    """Frontend using Tkinter"""
    
    def __init__(self, X, Y, unit, path, food):
        self.root = tk.Tk()
        self.X = X
        self.Y = Y
        self.unit = unit
        self.setup_gui(path, food)
    
    def setup_gui(self, path, food):
        """sets up the frame and playing world"""
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame, 
                                width=self.X*self.unit, 
                                height=self.Y*self.unit, 
                                bg='blue')
        self.canvas.pack()
        
        
        self.draw(path, food)
        
    def draw(self, path, food):
        """draws state on playing world
        path: list of coordinates of the snake body
        food: coordinate of the food
        """
        self.canvas.delete('all')
        
        for y,x in path:
            self.canvas.create_oval(
                    x*self.unit, y*self.unit, 
                    x*self.unit+self.unit, y*self.unit+self.unit, 
                    fill = 'red'
                    )
            
        fy, fx = food
        self.canvas.create_oval(
                fx*self.unit, fy*self.unit, 
                fx*self.unit+self.unit, fy*self.unit+self.unit, 
                fill='green'
                )
        
        
# Type alias for inforation that will be used by the neural network
# SnakeState = namedtuple('SnakeState', 'hunger array')       

class GameOfSnake:
    """Game logic for the game snake"""
    
    def __init__(self, path, food=None, X = 6, Y = 6, hasFrontEnd=True):
        
        self.turn = 0
        #self.orientation = Orientation.East # start of facing East
        self.turns_since_food = 0
        self.path = path
        self.head = path[-1]
        self.X = X
        self.Y = Y
        self.abyss = None # list of coordinates
        self.hunger = 100
        self.done = False
        self.food = food or self.new_apple_position()
        self.rewards = {'food': 1, 'dead': -1, 'survive': 1, 'abyss': -1, 'self_collide': -1, 'hunger': -1}
        
        self.frontend = FrontEnd(self.X, self.Y, 10, self.path, self.food) if hasFrontEnd else None
        
    def to_state(self):
        return self.hunger, self.array()
        
    def array(self):
        """creates and array of the playing world, where -1 is the location of 
        the food, and the snake body is encoded with floats between 1 (the head) and 0.5"""
        result = np.zeros((self.X, self.Y), dtype=float)
        l = len(self.path)
        for i, (x, y) in enumerate(self.path):
            result[x, y] = 0.5 + (i)*0.5/(l-1)# to make snake path values go down linearly, so that we can see its direction
            
        food_x, food_y = self.food
        result[food_x, food_y] = 0.2
        
        return self.add_peremiter(result)

    @staticmethod
    def add_peremiter(arr):
        """return same array but padded with -10s on all sides"""
        x, y = arr.shape
        result = np.full((x+2, y+2), -1, dtype=float)
        result[1:-1, 1:-1] = arr
        return result
        

    def new_apple_position(self):
        """select a new position for the food"""
        x = random.randint(0, self.X-1)
        y = random.randint(0, self.Y-1)

        if (x,y) in self.path:
            return self.new_apple_position()
        return (x,y)        

    def move_ahead(self, event):
        self.step(Direction.Ahead)
    def move_right(self, event):
        self.step(Direction.Right)
    def move_left(self, event):
        self.step(Direction.Left)
    
    def step(self, direction: Direction):
        """make a step, this method corresponds to the step method
        of openai gym environments"""
        
        new_head = self.update_head(direction)
        
        
        if self.done:
            print('game is already done')
            return None, 0, self.done
        
        self.turn += 1
            
        # check if no game over
        if new_head[0] < 0 or new_head[0] >= self.X or new_head[1] < 0 or new_head[1] >= self.Y:
            print('collided with perimeter')
            self.done=True
            return None, self.rewards['abyss'], self.done
        elif new_head in self.path:
            print('collided with self')
            self.done = True
            return None, self.rewards['self_collide'], self.done
        elif self.hunger < 1:
            self.done = True
            print('starved to death')
            return None, self.rewards['hunger'], self.done
        else:
            # still alive
            
            self.hunger -= 1
            self.path.append(new_head)
            self.head = new_head
            
            # check if it encounters food
            if self.food == new_head:
                self.turns_since_food = 0
                self.hunger += self.X * self.Y 
            
                self.food = self.new_apple_position()
                reward = self.rewards['food']        
            else:
                reward = 0
                self.turns_since_food += 1
                self.path.pop(0)   
            
            
        if self.frontend is not None:
            self.frontend.draw(self.path, self.food)
        
        return self.to_state(), reward, self.done

            
    def update_head(self, direction: Direction):
        #new_orientation = self.orientation.go(direction)
        new_head = tuple(np.array(self.head) + direction.vec())
        
        return new_head

   
    
    
def main():
    """create a game and bind events to the frontend, to play this game as-is."""
      
    game = GameOfSnake([(0,0), (0,1), (0,2)])
    
    game.frontend.root.bind('<Up>', game.move_ahead)
    #game.frontend.root.bind('<Down>', game.move_down)
    game.frontend.root.bind('<Right>', game.move_right)
    game.frontend.root.bind('<Left>', game.move_left)
    game.frontend.root.mainloop()
    
if __name__ == '__main__':
    """if this py-file is exectued directly"""
    main()
        
        
