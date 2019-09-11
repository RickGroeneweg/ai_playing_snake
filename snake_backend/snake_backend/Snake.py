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
        
    


class GameOfSnake:
    """Game logic for the game snake"""
    
    def __init__(self, path = None, food=None, X = 5, Y = 5, hasFrontEnd=True):
        
        self.turn = 0
        self.X = X
        self.Y = Y
        #self.orientation = Orientation.East # start of facing East
        self.turns_since_food = 0
        #print('set path')
        self.path = self.random_start_path()
        #print(f'path: {self.path}')
        self.head = self.path[-1]

        self.abyss = None # list of coordinates
        self.done = False
        self.food = food or self.new_apple_position()
        self.rewards = {'food': 100, 'dead': -5, 'survive': 0.1, 'abyss': -5, 'self_collide': -5}
        
        self.frontend = FrontEnd(self.X, self.Y, 10, self.path, self.food) if hasFrontEnd else None

    def random_square_next_to(self, x,y, excluding = []):
        x_, y_ =random.choice([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        
        if (x_, y_) in excluding:
            # try again
            return self.random_square_next_to(x,y, excluding=excluding)
        elif x_ < 0 or y_ <0 or x_ >self.X-1 or y_ >self.Y-1:
            # try again
            return self.random_square_next_to(x,y, excluding=excluding)
        else:
            return x_, y_

    def random_start_path(self):
        x_0 = random.randint(0, self.X-1)
        y_0 = random.randint(0, self.Y-1)
        
        x_1, y_1 = self.random_square_next_to(x_0, y_0, excluding = [(x_0, y_0)])
        x_2, y_2 = self.random_square_next_to(x_1, y_1, excluding = [(x_0, y_0), (x_1, y_1)])
        result = [(x_2, y_2), (x_1, y_1), (x_0, y_0)]
        #print(result)
        return result

        
    def array(self):
        """
        create and array of the playing world
        """
        # 3 chanels
        result = np.zeros((3, self.X, self.Y), dtype=float)
        
        for (x, y) in self.path:
            # [0]th chanel is the snake
            result[0, x, y] = 1 
            
        # [1]th chanel the head of the snake
        head_x, head_y = self.head
        result[1, head_x, head_y] = 1
            
        # [2]th chanel the food
        food_x, food_y = self.food
        result[2, food_x, food_y] = 1
        
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
        
        new_head = self.get_new_head(direction)
        
        
        if self.done:
            return None, 0, self.done
        
        self.turn += 1
            
        # check if no game over
        if new_head[0] < 0 or new_head[0] >= self.X or new_head[1] < 0 or new_head[1] >= self.Y:
            self.done=True
            return self.array(), self.rewards['abyss'], self.done
        elif new_head in self.path:
            self.done = True
            return self.array(), self.rewards['self_collide'], self.done
        else:
            # still alive
    
            self.path.append(new_head)
            self.head = new_head
            
            # check if it encounters food
            if self.food == new_head:
                self.turns_since_food = 0
            
                self.food = self.new_apple_position()
                reward = self.rewards['food']        
            else:
                reward = self.rewards['survive']
                self.turns_since_food += 1
                self.path.pop(0)   
            
            
        if self.frontend is not None:
            self.frontend.draw(self.path, self.food)
        
        return self.array(), reward, self.done

            
    def get_new_head(self, direction: Direction):
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
        
        
