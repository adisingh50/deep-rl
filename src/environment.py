
import pdb
import pickle
import time
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from blob import Blob

ENEMY_PENALTY = -300 
FOOD_REWARD = 300
MOVE_PENALTY = -1

class Environment:

    def __init__(self, grid_size, return_images):
        self.grid_size = grid_size
        self.return_images = return_images

    def reset(self):
        """Resets the environment by respawning the player, food, and enemy locations.

        Returns:
            state: the state of the newly initialized environment.
        """

        # Initialize new, random locations for the 3 blobs.
        self.player = Blob(grid_size=self.grid_size)

        self.food = Blob(grid_size=self.grid_size)
        while self.food == self.player:
            self.food = Blob(grid_size=self.grid_size)

        self.enemy = Blob(grid_size=self.grid_size)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(grid_size=self.grid_size)

        # Q-Agent observes state as x, y coordinate difference
        # Deep Q-Agnet observers state as the RGB image
        if self.return_images:
            state = self.get_image()
        else:
            state = (self.player - self.food, self.player - self.enemy)

        return state

    def step(self, action) -> Tuple[int, torch.Tensor, bool]:
        """Executes a player's action in the environment. 

        Args:
            action (int): the action ID between [0, 3] inclusive.

        Returns:
            reward: the agent's reward for the (state, action) pair.
            new_state: the new state for the agent.
            finished_episode: value determining if the agent completed the episode.
        """
        self.player.execute_action(action)

        # Get new state
        if self.return_images:
            new_state = self.get_image()
        else:
            new_state = (self.player - self.food, self.player - self.enemy)

        # Determine reward for (state, action) pair
        enemyCollision = self.player == self.enemy
        foodCollision = self.player == self.food

        if enemyCollision:
            reward = ENEMY_PENALTY
        elif foodCollision:
            reward = FOOD_REWARD
        else:
            reward = MOVE_PENALTY

        finished_episode = enemyCollision or foodCollision

        return reward, new_state, finished_episode


    def display_game(self, episode) -> None:
        """Enlarges and displays the environment using OpenCV visual interface.

        Args:
            episode (int): The current episode we are training on.
        """
        env = self.get_image().permute((1, 2, 0)) # convert (C, H, W) -> (H, W, C)
        env = env.detach().numpy()
        img = cv2.resize(env, (300,300), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"Adi Game - Episode {episode}", img)
        cv2.waitKey(1)
        time.sleep(0.2)

    def get_image(self):
        env = torch.zeros((3, self.grid_size, self.grid_size))
        env[:, self.player.y, self.player.x] = torch.Tensor([0, 0, 255])
        env[:, self.food.y, self.food.x] = torch.Tensor([0, 255, 0])
        env[:, self.enemy.y, self.enemy.x] = torch.Tensor([255, 0, 0])
        return env

