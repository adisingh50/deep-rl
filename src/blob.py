import numpy as np

class Blob:
    def __init__(self, grid_size):

        # Initialize a unique pair of coordinates for the blob.
        self.x = np.random.randint(0, grid_size)
        self.y = np.random.randint(0, grid_size)

        self.grid_size = grid_size
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y)

    def execute_action(self, choice):
        if (choice == 0):
            self.move(deltaX=0, deltaY=1)
        elif (choice == 1):
            self.move(deltaX=1, deltaY=0)
        elif (choice == 2):
            self.move(deltaX=0, deltaY=-1)
        elif (choice == 3):
            self.move(deltaX=-1, deltaY=0)

    def move(self, deltaX, deltaY):

        self.x += deltaX
        self.y += deltaY

        # Keep player within bounds of the grid.
        if self.x < 0:
            self.x = 0
        elif self.x > self.grid_size-1:
            self.x = self.grid_size-1
        
        if self.y < 0:
            self.y = 0
        elif self.y > self.grid_size-1:
            self.y = self.grid_size-1

        

