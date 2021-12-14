
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

from blob import Blob

# Constants for custom game environment
EPISODES = 25000
ENEMY_PENALTY = 300
FOOD_REWARD = 25
EPSILON = 0.9
EPSILON_DECAY = 0.9998
SHOW_EVERY = 3000

alpha = 0.1
discount = 0.95
