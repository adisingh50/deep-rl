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

    def __init__(self,
                 grid_size, 
                 replay_memory_size, 
                 min_replay_memory_size, 
                 batch_size, gamma, 
                 target_model_update_interval, 
                 epsilon, 
                 epsilon_decay, 
                 min_epsilon):
        self.env = Environment(grid_size=grid_size, return_images=True)
        self.pred_model = Encoder(action_space_size=4)
        self.target_model = Encoder(action_space_size=4)
        self.target_model.load_state_dict(self.pred_model.state_dict())
        self.pred_model.train()
        self.target_model.eval()

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.min_replay_memory_size = min_replay_memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_model_update_interval = target_model_update_interval
        self.target_update_counter = 0
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.huber_loss = nn.SmoothL1Loss().to(self.pred_model.device)
        self.optimizer = torch.optim.SGD(self.pred_model.parameters(), lr=0.01)

        self.losses = []
        self.episode_rewards = []
        self.step_counts = []
        self.total_steps = 0

        self.modelID = int(time.time())
        os.makedirs(f"results/deep-qagent-{self.modelID}", exist_ok=True)

    def engage_environment(self, num_episodes):

        for episode in range(num_episodes):
            episode_reward = 0
            steps = 0
            current_state = self.env.reset()

            done = False
            while not done:
                if steps >= 1000: # agent is getting lost, just truncate the episode
                    break

                # e-greedy approach to determine action
                if np.random.random() > self.epsilon: #exploitation
                    state_tensor = torch.unsqueeze(current_state, dim=0).to(self.pred_model.device)
                    action = torch.argmax(self.pred_model.forward(state_tensor)).item()
                else: #exploration
                    action = np.random.randint(0, 4)

                # Make a step in the environment
                reward, new_state, done = self.env.step(action)
                episode_reward += reward

                # Update replay memory and train prediction model on a minibatch
                dataTuple = (current_state, action, reward, new_state, done)
                self.update_replay_memory(dataTuple)
                self.train_minibatch(done)

                current_state = new_state
                steps += 1

            self.total_steps += steps
            self.episode_rewards.append(episode_reward)
            self.step_counts.append(steps)

            # Print out training metrics on some episodes.
            if episode > 0:
                print(f"On Episode: {episode} | Epsilon: {self.epsilon} | Reward: {self.episode_rewards[-1]} | Steps: {self.step_counts[-1]}")

            # Epsilon Decay
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.min_epsilon)

            # Save model every 500 episodes
            if episode > 0 and episode % 500 == 0:
                torch.save(self.pred_model.state_dict(), f"results/deep-qagent-{self.modelID}/model-{episode}-episodes.pt")


    def train_minibatch(self, terminal_state) -> None:
        self.optimizer.zero_grad()

        # Can only begin training after agent has made a certain number of steps.
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        minibatch = random.sample(self.replay_memory, self.batch_size)# [(s,a,r,s',d)1, (s,a,r,s',d)2, ...]

        # Force the most recent transition tuple to be in the minibatch
        # minibatch[-1] = self.replay_memory[-1]

        # Obtain batches for state, action, reward and new state
        s_a_r_s_d = list(zip(*minibatch))

        state_batch = (torch.stack(s_a_r_s_d[0]) / 255.0).to(self.pred_model.device) # shape: (N,C,H,W)
        action_batch = torch.Tensor(s_a_r_s_d[1]).to(torch.int64).to(self.pred_model.device) # shape: (N,)
        reward_batch = torch.Tensor(s_a_r_s_d[2]).unsqueeze(dim=1).to(self.pred_model.device) # shape: (N,1)
        new_state_batch = (torch.stack(s_a_r_s_d[3]) / 255.0).to(self.pred_model.device) # shape: (N,C,H,W)
        done_batch = torch.Tensor(s_a_r_s_d[4]).to(torch.bool) # shape: (N, )

        Q_pred = self.pred_model.forward(state_batch).gather(1, action_batch.unsqueeze(1)) # shape: (N,1)
        max_future_q_target = self.target_model.forward(new_state_batch).max(dim=1)[0].unsqueeze(dim=1)
        Q_target = reward_batch + self.gamma*max_future_q_target # shape: (N,1)

        # For (state, action) pairs which terminated the episode, overwrite their Q-values with the reward 
        Q_target[done_batch] = reward_batch[done_batch] 

        # Compute loss and update paramaters of pred_model
        loss = self.huber_loss(Q_pred, Q_target)
        loss.backward()
        self.losses.append(loss)
        self.optimizer.step()

        # Increment counter if we finished an episode and copy weights from pred_model -> target_model if needed.
        if terminal_state:
            self.target_update_counter += 1

            if self.target_update_counter >= self.target_model_update_interval:
                print("Copying Pred Model Weights over to Target Model...")
                print("Saving updated loss plot...")
                self.target_model.load_state_dict(self.pred_model.state_dict())
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

        plt.scatter(x_values, y_values)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Deep RL Agent Losses")
        plt.savefig(f"results/deep-qagent-{self.modelID}/loss_plot_{self.total_steps}_steps.png")
        plt.clf()

    def save_results_to_disk(self, window_size):
        print("Saving Deep-Q-Model and Episode Reward Metrics to Disk...")

        # Save Prediction Network
        torch.save(self.pred_model.state_dict(), f"results/deep-qagent-{self.modelID}/model-final.pt")

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
    agent = DeepQAgent(grid_size=10,
                    replay_memory_size=10, 
                    min_replay_memory_size=5, 
                    batch_size=4, 
                    gamma=0.99, 
                    target_model_update_interval=10,
                    epsilon=0.99,
                    epsilon_decay=0.995,
                    min_epsilon=0.05
                )

    # Fill dummy data for agent's replay memory
    for i in range(10):
        state = torch.rand(3, 10, 10)
        action = np.random.randint(0,4)
        reward = np.random.randint(-10,10)
        new_state = torch.rand(3, 10, 10)

        dataTuple = (state, action, reward, new_state, (i % 2 == 0))
        agent.update_replay_memory(dataTuple)

    agent.train_minibatch(False)

    