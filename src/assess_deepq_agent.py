"""Testing script that loads in a Pytorch DQN and visualizes an agent that obeys the Q-network."""

import pdb

import cv2
import numpy as np
import torch

from encoder import Encoder
from environment import Environment

if __name__ == "__main__":
    PATH = "results/deep-qagent-1640108893/model-final.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load Pytorch DQN
    Q_network = Encoder(action_space_size=4)
    Q_network.load_state_dict(torch.load(PATH, map_location=device))

    env = Environment(grid_size=10, return_images=True)
    state = env.reset()
    episode_reward = 0

    done = False
    while not done:
        # Display the current game state using OpenCV
        env.display_game(1)

        # Step in the environment
        state_tensor = torch.unsqueeze(state, dim=0)
        action = torch.argmax(Q_network.forward(state_tensor)).item()
        reward, new_state, done = env.step(action)

        episode_reward += reward
        state = new_state
    
    cv2.destroyAllWindows()
    print(f"Final Reward: {episode_reward}")

