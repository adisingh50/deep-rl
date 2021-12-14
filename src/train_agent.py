
from environment import Environment

env = Environment(alpha=0.1, grid_size=10, episodes=25000, num_steps=500, epsilon=0.9, epsilon_decay=0.9998)

env.train()
env.plot_episode_rewards()
env.save_q_table()
print('Done Training RL Agent :)')