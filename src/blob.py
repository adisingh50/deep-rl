import numpy as np

class Blob:
    def __init__(self, grid_size, exclude_coords=()):

        # Initialize a unique pair of coordinates for the blob.
        while True:
            self.x = np.random.randint(0, grid_size)
            self.y = np.random.randint(0, grid_size)

            coords = (self.x, self.y)
            if coords not in exclude_coords:
                break

        self.grid_size = grid_size
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def execute_action(self, choice):
        if (choice == 0):
            self.move(deltaX=0, deltaY=1)
        elif (choice == 1):
            self.move(deltaX=1, deltaY=0)
        elif (choice == 2):
            self.move(deltaX=0, deltaY=-1)
        elif (choice == 3):
            self.move(deltaX=-1, deltaY=0)

    def move(self, deltaX=False, deltaY=False):

        if not deltaX and not deltaY:
            moveID = np.random.randint(0,4)

            if (moveID == 0):
                self.y -= 1
            elif (moveID == 1):
                self.x += 1
            elif (moveID == 2):
                self.y += 1
            elif (moveID == 3):
                self.x -= 1
        else:
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

        

