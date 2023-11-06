import random
import numpy as np


random.seed(2)
pos = np.array([np.array([random.uniform(a=-0.35, b=0.15), random.uniform(a=-0.4, b=0.1)]) for i in range(10)])

pos_in_real = pos * 1000
pos_in_real[:, 0] += 463.9
print(pos)
print(pos_in_real)