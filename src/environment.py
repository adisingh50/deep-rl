
import cv2
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from blob import Blob

ENEMY_PENALTY = 300
FOOD_REWARD = 25
MOVE_PENALTY = 1
SHOW_EVERY = 3000

class Environment:

    def __init__(self, alpha, grid_size, episodes, num_steps, epsilon, epsilon_decay):
        self.alpha = alpha
        self.grid_size = grid_size
        self.episodes = episodes
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize entities
        self.player, self.food, self.enemy = self.create_entities()
        self.episode_rewards = []

        # Initialize q-table.
        # Maps ((food deltaX, food deltaY), (enemy deltaX, enemy deltaY)) -> reward
        self.q_table = {}

        self.action_space = (0, 1, 2, 3)
        for food_deltaX in range(-grid_size+1, grid_size):
            for food_deltaY in range(-grid_size+1, grid_size):
                for enemy_deltaX in range(-grid_size+1, grid_size):
                    for enemy_deltaY in range(-grid_size+1, grid_size):
                            self.q_table[((food_deltaX, food_deltaY), (enemy_deltaX, enemy_deltaY))] = [np.random.random() for i in range(4)]
        
    
    def train(self):
        for episode in range(self.episodes):

            if episode != 0 and episode % SHOW_EVERY == 0:
                print(f"On Episode: {episode} | Epsilon: {self.epsilon} | Reward: {self.episode_rewards[-1]}")
                show_display = True
            else:
                show_display = False
            

            episode_reward = 0
            for i in range(self.num_steps):
                obs = (self.player - self.food, self.player - self.enemy)

                if np.random.random() > self.epsilon: # exploitation
                    action = np.argmax(self.q_table[obs])
                else:                                 # exploration
                    action = np.random.randint(0, 4)
                self.player.execute_action(action)

                # Determine reward for (state, action) pair
                if self.player.x == self.enemy.x and self.player.y == self.enemy.y:
                    reward = -ENEMY_PENALTY
                elif self.player.x == self.food.x and self.player.y == self.food.y:
                    reward = FOOD_REWARD
                else:
                    reward = -MOVE_PENALTY
                
                new_obs = (self.player - self.food, self.player - self.enemy)
                max_future_q = np.max(self.q_table[new_obs])
                curr_q = self.q_table[obs][action]

                # Update Q value
                if reward == FOOD_REWARD:
                    new_q = FOOD_REWARD
                elif reward == -ENEMY_PENALTY:
                    new_q = -ENEMY_PENALTY
                else:
                    new_q = (1 - self.alpha)*(curr_q) + (self.alpha)*(reward + self.epsilon_decay*max_future_q)
                self.q_table[obs][action] = new_q

                # Display Game UI on occasional episodes
                if show_display:
                    self.display_game(episode)

                # Update episode reward and potentially end the episode.
                episode_reward += reward
                if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                    break
            
            self.episode_rewards.append(episode_reward)
            self.epsilon *= self.epsilon_decay

    def display_game(self, episode):
        env = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        env[self.player.y][self.player.x] = (255, 0, 0)
        env[self.food.y][self.food.x] = (0, 255, 0)
        env[self.enemy.y][self.enemy.x] = (0, 0, 255)

        img = Image.fromarray(env, "RGB")
        img = img.resize((300, 300))
        cv2.imshow(f"Adi Game - Episode{episode}", np.array(img))
        time.sleep(0.5)


    def create_entities(self):
        player = Blob()
        playerCoords = (player.x, player.y)

        food = Blob(exclude_coords=playerCoords)
        foodCoords = (food.x, food.y)

        playerAndFood = (playerCoords, foodCoords)
        enemy = Blob(exclude_coords=playerAndFood)

        return player, food, enemy

    def plot_episode_rewards(self):
        x_values = np.arange(0, len(self.episode_rewards))
        y_values = np.array(self.episode_rewards)
        
        plt.plot(x_values, y_values)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("RL Agent Episode Rewards")
        plt.savefig("reward_plot.png")

    def save_q_table(self):
        with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
            pickle.dump(self.q_table)