import gym
import numpy as np

import mani_skill2.envs 


env = gym.make(id="PegInsertionSide-v4",
               control_mode='pd_ee_vel_pos',)
obs = env.reset()
env.step(np.array([0.0, 0.0, 0.0, 0.0]))