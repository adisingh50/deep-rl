
import cv2
import numpy as np

from q_agent import QAgent
from deep_q_agent import DeepQAgent

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

def train_deep_q_agent():
    deepQAgent = DeepQAgent(grid_size=10,
                            replay_memory_size=5000,
                            min_replay_memory_size=500,
                            batch_size=64,
                            gamma=0.999,
                            target_model_update_interval=10,
                            epsilon=0.99,
                            epsilon_decay=0.9997,
                            min_epsilon=0.001)
    deepQAgent.engage_environment(num_episodes=10_000)
    deepQAgent.save_results_to_disk(window_size=50)
    print('Done Training Deep RL Agent :)')

if __name__ == "__main__":
    train_deep_q_agent()