"""Class that trains a Q Learning Agent."""

import os
import pdb
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from environment import Environment

ENEMY_PENALTY = -300
FOOD_REWARD = 300
MOVE_PENALTY = -1
SHOW_EVERY = 2500

class QAgent:
    def __init__(self, alpha, gamma, grid_size, episodes, num_steps, epsilon, epsilon_decay):
        self.alpha = alpha
        self.gamma = gamma
        self.grid_size = grid_size
        self.episodes = episodes
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.episode_rewards = []

        # Initialize environment and q-table.
        # Maps ((food deltaX, food deltaY), (enemy deltaX, enemy deltaY)) -> reward
        self.env = Environment(self.grid_size, return_images=True)
        self.q_table = {}

        self.action_space = (0, 1, 2, 3)
        for food_deltaX in range(-grid_size+1, grid_size):
            for food_deltaY in range(-grid_size+1, grid_size):
                for enemy_deltaX in range(-grid_size+1, grid_size):
                    for enemy_deltaY in range(-grid_size+1, grid_size):
                            self.q_table[((food_deltaX, food_deltaY), (enemy_deltaX, enemy_deltaY))] = [np.random.random() for i in range(4)]
    

    def train(self):
        # Iterate through all episodes
        for episode in range(self.episodes):
            self.env.reset()

            # Print out training metrics on some episodes.
            if episode != 0 and episode % SHOW_EVERY == 0:
                print(f"On Episode: {episode} | Epsilon: {self.epsilon} | Reward: {self.episode_rewards[-1]}")
                show_display = False
            else:
                show_display = False
            

            episode_reward = 0
            # Iterate through all steps within episode
            for i in range(self.num_steps):

                # Display Game UI on occasional episodes
                if show_display:
                    self.env.display_game(episode)

                # Get (state, action) pair
                obs = (self.env.player - self.env.food, self.env.player - self.env.enemy)
                if np.random.random() > self.epsilon: # exploitation
                    action = np.argmax(self.q_table[obs])
                else:                                 # exploration
                    action = np.random.randint(0, 4)
                self.env.player.execute_action(action)

                # Determine reward for (state, action) pair
                enemyCollision = self.env.player.x == self.env.enemy.x and self.env.player.y == self.env.enemy.y
                foodCollision = self.env.player.x == self.env.food.x and self.env.player.y == self.env.food.y

                if enemyCollision:
                    reward = ENEMY_PENALTY
                elif foodCollision:
                    reward = FOOD_REWARD
                else:
                    reward = MOVE_PENALTY
                
                new_obs = (self.env.player - self.env.food, self.env.player - self.env.enemy)
                max_future_q = np.max(self.q_table[new_obs])
                curr_q = self.q_table[obs][action]

                # Update Q value
                if reward == FOOD_REWARD:
                    new_q = FOOD_REWARD
                elif reward == ENEMY_PENALTY:
                    new_q = ENEMY_PENALTY
                else:
                    new_q = (1 - self.alpha)*(curr_q) + (self.alpha)*(reward + self.gamma*max_future_q)
                self.q_table[obs][action] = new_q

                # Update episode reward and potentially end the episode.
                episode_reward += reward
                if foodCollision or enemyCollision:
                    break
            
            # If gameplay was displayed for an episode, kill the window
            if show_display:
                cv2.destroyAllWindows()
            
            self.episode_rewards.append(episode_reward)
            self.epsilon *= self.epsilon_decay


    def save_results_to_disk(self, window_size):
        print("Saving Q-Table and Episode Reward Metrics to Disk...")
        currTime = int(time.time())

        # Save qtable dictionary
        os.makedirs(f"results/qagent-{currTime}", exist_ok=True)
        with open(f"results/qagent-{currTime}/qtable-{currTime}.pickle", "wb") as f:
            pickle.dump(self.q_table, f)

        # Write all episode rewards to txt
        with open(f"results/qagent-{currTime}/episode-rewards.txt", "a") as f:
            for episode, reward in enumerate(self.episode_rewards):
                f.write(f"Episode: {episode} | Reward: {reward}\n")

        # Plot Episode Rewards
        moving_avg = np.convolve(self.episode_rewards, np.ones((window_size,)) / window_size, mode='valid')

        x_values = np.arange(SHOW_EVERY, SHOW_EVERY + len(moving_avg))
        y_values = np.array(moving_avg)
        
        plt.plot(x_values, y_values)
        plt.xlabel("Episode")
        plt.ylabel("Reward Moving Average")
        plt.title("RL Agent Episode Rewards")
        plt.savefig(f"results/qagent-{currTime}/reward_plot.png")

