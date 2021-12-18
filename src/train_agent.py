
import cv2
import numpy as np

from q_agent import QAgent

def display_random_moves():
    env = Environment(alpha=0.1, grid_size=10, episodes=25000, num_steps=500, epsilon=0.9, epsilon_decay=0.9998)

    while True:
        random_action = np.random.randint(0, 4)
        env.player.execute_action(random_action)
        terminate = env.display_game()

        if terminate:
            break

def train_q_agent():
    qAgent = QAgent(alpha=0.15, gamma=0.95, grid_size=10, episodes=400000, num_steps=500, epsilon=0.9, epsilon_decay=0.999992)
    qAgent.train()
    qAgent.save_results_to_disk(window_size=50)
    print('Done Training RL Agent :)')

if __name__ == "__main__":
    train_q_agent()