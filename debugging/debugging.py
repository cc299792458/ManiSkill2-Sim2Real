import gym
import numpy as np

import mani_skill2.envs 


env = gym.make(id="PickCube-v3",
               control_mode='constvel_ee_delta_pos',)
obs = env.reset()
env.step(np.array([0.0, 0.0, 0.0, 0.0]))