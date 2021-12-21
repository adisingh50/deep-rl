import pickle

import cv2
import numpy as np

from q_agent import QAgent
from deep_q_agent import DeepQAgent

from environment import Environment

def take_random_moves():
    env = Environment(grid_size=10, return_images=True)
    state = env.reset()
    episode_reward = 0
    batch = []

    done = False
    while not done:
        env.display_game(1)
        action = np.random.randint(0, 4)
        reward, new_state, done = env.step(action)

        transition = (state, action, reward, new_state, done)
        batch.append(transition)

        episode_reward += reward
        state = new_state

    with open("minibatch.pickle", "wb") as f:
        pickle.dump(batch, f)

    cv2.destroyAllWindows()
    print(f"Final Reward: {episode_reward}")

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
                            gamma=0.99,
                            target_model_update_interval=10,
                            epsilon=1.0,
                            epsilon_decay=0.995,
                            min_epsilon=0.001,
                            lr=0.001)
    deepQAgent.engage_environment(num_episodes=1000)
    deepQAgent.save_results_to_disk(window_size=50)
    print('Done Training Deep RL Agent :)')

if __name__ == "__main__":
    take_random_moves()