import gym
import numpy as np

import mani_skill2.envs 


env = gym.make(id="PegInsertionSide2D-v0",
               control_mode='pd_ee_vel_pos',)
obs = env.reset()
while True:
    env.render(mode="human")
    env.step(np.array([0.0, 0.0, 0.0, 1.0]))