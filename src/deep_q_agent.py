"""Class that creates a Deep Q Learning Agent."""

import copy
import glob
import os
import pdb
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder
from environment import Environment

ENEMY_PENALTY = -300
FOOD_REWARD = 300
MOVE_PENALTY = -1
MAX_NUM_STEPS_PER_EPISODE = 5000
SHOW_EVERY = 2500

class DeepQAgent:

    def __init__(self, grid_size, replay_memory_size, min_replay_memory_size, batch_size, gamma, target_model_update_interval, epsilon, epsilon_decay, min_epsilon):
        self.env = Environment(grid_size=grid_size, return_images=True)
        self.pred_model = Encoder(action_space_size=4)
        self.target_model = copy.deepcopy(self.pred_model)

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.min_replay_memory_size = min_replay_memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_model_update_interval = target_model_update_interval
        self.target_update_counter = 0
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pred_model.parameters(), lr=0.01)

        if torch.cuda.is_available():
            self.pred_model = self.pred_model.cuda()
            self.target_model = self.target_model.cuda()
            self.mse_loss = self.mse_loss.cuda()

        self.losses = []
        self.episode_rewards = []
        self.step_counts = []
        self.total_steps = 0

        self.modelID = int(time.time())
        os.makedirs(f"results/deep-qagent-{self.modelID}", exist_ok=True)

    def engage_environment(self, num_episodes):
        self.pred_model.train()

        for episode in range(num_episodes):
            episode_reward = 0
            steps = 0
            current_state = self.env.reset()

            done = False
            while not done:
                # e-greedy approach to determine action
                if np.random.random() > self.epsilon: #exploitation
                    state_tensor = torch.unsqueeze(current_state, dim=0)
                    if torch.cuda.is_available():
                        state_tensor = state_tensor.cuda()

                    action = torch.argmax(self.pred_model.forward(state_tensor))
                else: #exploration
                    action = np.random.randint(0, 4)

                # Make a step in the environment
                reward, new_state, done = self.env.step(action)
                episode_reward += reward

                # Update replay memory and train the prediction model for every step
                dataTuple = (current_state, action, reward, new_state, done)
                self.update_replay_memory(dataTuple)
                self.train_minibatch(done)

                current_state = new_state
                steps += 1
                self.total_steps += 1

            self.episode_rewards.append(episode_reward)
            self.step_counts.append(steps)

            # Print out training metrics on some episodes.
            if episode > 0:
                print(f"On Episode: {episode} | Epsilon: {self.epsilon} | Reward: {self.episode_rewards[-1]} | Steps: {self.step_counts[-1]}")

            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.min_epsilon)


    def train_minibatch(self, terminal_state) -> None:
        self.optimizer.zero_grad()

        # Can only begin training after agent has made a certain number of steps.
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, self.batch_size)# [(s,a,r,s',d)1, (s,a,r,s',d)2, ...]
        current_states = torch.stack([dataTuple[0] for dataTuple in minibatch]) / 255.0 # shape: (N,C,H,W)
        if torch.cuda.is_available():
            current_states = current_states.cuda()

        Q_state_pred = self.pred_model.forward(current_states) # shape: (N, 4)
        Q_state_gt = torch.clone(Q_state_pred)

        new_states = torch.stack([dataTuple[3] for dataTuple in minibatch]) / 255.0 # shape: (N,C,H,W)
        if torch.cuda.is_available():
            new_states = new_states.cuda()

        Q_new_state_target = self.target_model.forward(new_states) # shape: (N, 4)

        # Iterate through minibatch to update Q_state_gt with "ground truth" Q-values
        for image_index, (state, action, reward, new_state, done) in enumerate(minibatch):

            # Determine the new q value for the (state, action) pair
            if done:
                new_q = reward
            else:
                max_future_q = torch.max(Q_new_state_target[image_index])
                new_q = reward + self.gamma*max_future_q
            
            # Update Q value for the given state
            Q_state_gt[image_index, action] = new_q

        # Update paramaters of pred_model
        loss = self.mse_loss(Q_state_pred, Q_state_gt)
        loss.backward()
        self.losses.append(loss)
        self.optimizer.step()

        # Increment counter if we finished an episode and copy weights from pred_model -> target_model if needed.
        if terminal_state:
            self.target_update_counter += 1

            if self.target_update_counter >= self.target_model_update_interval:
                print("Copying Pred Model Weights over to Target Model...")
                print("Saving updated loss and reward plots...")
                self.target_model = copy.deepcopy(self.pred_model)
                self.target_update_counter = 0

                self.save_loss_plot()

    def update_replay_memory(self, dataTuple):
        self.replay_memory.append(dataTuple)

    def save_loss_plot(self):
        # Delete an old loss plot file if it exists
        lossPlotPaths = glob.glob(f"/content/drive/My Drive/deep-rl/results/deep-qagent-{self.modelID}/loss_plot*.png")
        if len(lossPlotPaths) == 1 and os.path.exists(lossPlotPaths[0]):
            os.remove(lossPlotPaths[0])

        x_values = np.arange(self.min_replay_memory_size, self.min_replay_memory_size + len(self.losses))
        y_values = np.array(self.losses)

        plt.plot(x_values, y_values)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Deep RL Agent Losses")
        plt.savefig(f"results/deep-qagent-{self.modelID}/loss_plot_{self.total_steps}_steps.png")
        plt.clf()

    def save_results_to_disk(self, window_size):
        print("Saving Deep-Q-Model and Episode Reward Metrics to Disk...")

        # Save Prediction Network
        torch.save(self.pred_model.state_dict(), f"results/deep-qagent-{self.modelID}/model.pt")

        # Write all episode rewards and losses to txt
        with open(f"results/deep-qagent-{self.modelID}/episode-rewards.txt", "a") as f:
            for episode, reward in enumerate(self.episode_rewards):
                f.write(f"Episode: {episode} | Reward: {reward}\n")

        with open(f"results/deep-qagent-{self.modelID}/step-losses.txt", "a") as f:
            for step, loss in enumerate(self.losses):
                f.write(f"Step: {step + self.min_replay_memory_size} | Loss: {loss}\n")

        # Plot Episode Rewards Plot
        moving_avg = np.convolve(self.episode_rewards, np.ones((window_size,)) / window_size, mode='valid')

        x_values = np.arange(window_size, window_size + len(moving_avg))
        y_values = np.array(moving_avg)

        plt.plot(x_values, y_values)
        plt.xlabel("Episode")
        plt.ylabel("Reward Moving Average")
        plt.title("Deep RL Agent Episode Rewards")
        plt.savefig(f"results/deep-qagent-{self.modelID}/reward_plot.png")
        plt.clf()

        # Plot Losses Plot
        self.save_loss_plot()
        

if __name__ == "__main__":
    agent = DeepQAgent(replay_memory_size=10, min_replay_memory_size=5, batch_size=3, gamma=0.95, target_model_update_interval=5)

    # Fill dummy data for agent's replay memory
    for i in range(10):
        state = torch.rand(3, 10, 10)
        action = np.random.randint(0,4)
        reward = np.random.randint(-10,10)
        new_state = torch.rand(3, 10, 10)

        dataTuple = (state, action, reward, new_state, False)
        agent.update_replay_memory(dataTuple)

    agent.train(False)

    