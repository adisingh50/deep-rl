"""Testing Script that loads in a policy dictionary and visualizes an agent that obeys the policy."""

import cv2
import pdb
import pickle

import numpy as np

from environment import Environment

if __name__ == "__main__":
    PATH = "results/qagent-1639702949/qtable-1639702949.pickle"
    ENEMY_PENALTY = -300
    FOOD_REWARD = 300
    MOVE_PENALTY = -1

    # Load Q-table file
    with open(PATH, "rb") as f:
        qtable = pickle.load(f)

    env = Environment(grid_size=10)
    player, food, enemy = env.create_entities()

    reward = 0
    while True:
        # Display the current game state using OpenCV
        env.display_game(1, player, food, enemy)

        # Get (state, action) pair
        state = (player - food, player - enemy)
        action = np.argmax(qtable[state])
        player.execute_action(action)

        # Determine reward for (state, action) pair
        enemyCollision = player.x == enemy.x and player.y == enemy.y
        foodCollision = player.x == food.x and player.y == food.y

        # Update reward for the episode
        if enemyCollision:
            reward += ENEMY_PENALTY
        elif foodCollision:
            reward += FOOD_REWARD
        else:
            reward += MOVE_PENALTY

        if enemyCollision or foodCollision:
            break
    
    cv2.destroyAllWindows()
    print(f"Final Reward: {reward}")
    
    
    