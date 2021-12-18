
import pdb
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from blob import Blob

class Environment:

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def create_entities(self):
        player = Blob(grid_size=self.grid_size)
        playerCoords = (player.x, player.y)

        food = Blob(exclude_coords=playerCoords, grid_size=self.grid_size)
        foodCoords = (food.x, food.y)

        playerAndFood = (playerCoords, foodCoords)
        enemy = Blob(exclude_coords=playerAndFood, grid_size=self.grid_size)

        return player, food, enemy

    def display_game(self, episode, player, food, enemy):
        env = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        env[player.y, player.x] = (255, 0, 0)
        env[food.y, food.x] = (0, 255, 0)
        env[enemy.y,enemy.x] = (0, 0, 255)

        img = cv2.resize(env, (300,300), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"Adi Game - Episode {episode}", img)
        cv2.waitKey(1)
        time.sleep(0.2)
