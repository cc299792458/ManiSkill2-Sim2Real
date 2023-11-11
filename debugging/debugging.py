import gym
import numpy as np

import mani_skill2.envs 


env = gym.make(id="PegInsertionSide2D-v2",
               control_mode='pd_ee_vel_xy',)
obs = env.reset()
while True:
    env.render(mode="human")
    env.step(np.array([0.0, 0.0, 1.0]))