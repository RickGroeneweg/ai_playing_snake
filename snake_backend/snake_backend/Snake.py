# -*- coding: utf-8 -*-
"""
Frontend and Backend for the game snake, this file can be played as is by executing it with python, or it can be used by a different front end or in our case of interest, an openai gym environment.
"""

import tkinter as tk
import numpy as np
import random

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
        
        for x,y in path:
            self.canvas.create_oval(
                    x*self.unit, y*self.unit, 
                    x*self.unit+self.unit, y*self.unit+self.unit, 
                    fill = 'red'
                    )
            
        fx, fy = food
        self.canvas.create_oval(
                fx*self.unit, fy*self.unit, 
                fx*self.unit+self.unit, fy*self.unit+self.unit, 
                fill='green'
                )
        
        
        

class GameOfSnake:
    """Game logic for the game snake"""
    
    def __init__(self, path, food=None, X = 6, Y = 6, hasFrontEnd=True):
        
        self.turn = 0
        self.turns_since_food = 0
        self.path = path
        self.head = path[-1]
        self.X = X
        self.Y = Y
        self.done = False
        self.food = food or self.new_apple_position()
        self.rewards = {'food': 100, 'dead': -100, 'survive': 1}
        
        self.frontend = FrontEnd(self.X, self.Y, 10, self.path, self.food) if hasFrontEnd else None
        
    def array(self):
        """creates and array of the playing world, where -1 is the location of 
        the food, and the snake body is encoded with floats between 1 (the head) and 0.5"""
        result = np.zeros((self.X, self.Y), dtype=float)
        l = len(self.path)
        for i, (x, y) in enumerate(self.path):
            result[x, y] = 0.5 + (i)*0.5/(l-1)# to make snake path values go down linearly
            
        food_x, food_y = self.food
        result[food_x, food_y] = -1
        
        return result
        

    def new_apple_position(self):
        """select a new position for the food"""
        x = random.randint(0, self.X-1)
        y = random.randint(0, self.Y-1)

        if (x,y) in self.path:
            return self.new_apple_position()
        return (x,y)        

    def move_up(self, event):
        self.step((0, -1))
    def move_down(self, event):
        self.step((0, 1))
    def move_right(self, event):
        self.step((1, 0))
    def move_left(self, event):
        self.step((-1, 0))
    
    def step(self, direction):
        """make a step, this method corresponds to the step method
        of openai gym environments"""
        reward = 0
        new_head = self.add_vec(self.head, direction)
        
        
        if self.done:
            print('game is already done')
            return None, 0, self.done
        
        self.turn += 1
            
        # check if no game over
        if new_head[0] < 0 or new_head[0] >= self.X or new_head[1] < 0 or new_head[1] >= self.Y or new_head in self.path:
            print(f'game over, turn {self.turn}, length {len(self.path)}')
            self.done = True
            reward += self.rewards['dead']
            return self.array(), reward, self.done
        elif self.turns_since_food > 50*len(self.path):
            #no energy
            self.done = True
        else:
            # still alive
            
            reward += len(self.path)
            self.path.append(new_head)
            self.head = new_head
            
            # check if it encounters food
            if self.food == new_head:
                self.turns_since_food = 0
            
                self.food = self.new_apple_position()
                reward += self.rewards['food']        
            else:
                self.turns_since_food += 1
                self.path.pop(0)   
            
            
        if self.frontend is not None:
            self.frontend.draw(self.path, self.food)
        
        return self.array(), reward, self.done

            
    @staticmethod
    def add_vec(tuple1, tuple2):
        """add two tuples as if they are vectors"""
        a = tuple1[0] + tuple2[0]
        b = tuple1[1] + tuple2[1]
        return (a,b)
        
   
    
    
def main():
    """create a game and bind events to the frontend, to play this game as-is."""
      
    game = GameOfSnake([(0,0), (0,1), (0,2)])
    
    game.frontend.root.bind('<Up>', game.move_up)
    game.frontend.root.bind('<Down>', game.move_down)
    game.frontend.root.bind('<Right>', game.move_right)
    game.frontend.root.bind('<Left>', game.move_left)
    game.frontend.root.mainloop()
    
if __name__ == '__main__':
    """if this py-file is exectued directly"""
    main()
        
        
