# TODO(chichu): need to be removed
import sys
sys.path.append('/home/chichu/Documents/Sapien/ManiSkill2-Sim2Real')

import argparse
import os.path as osp
import gym
import numpy as np

from stable_baselines3 import PPO
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.utils.generate_sim_params import generate_sim_params
from train_eval.ppo_state import ContinuousTaskWrapper, SuccessInfoWrapper

from real_robot.agents.xarm import XArm7

class RealRobotEval():
    def __init__(self, env, model,
                 xarm_ip='192.168.1.229',
                 real_control_mode='pd_ee_delta_pose_axangle',
                 robot_action_scale=100,
                 enable_real_robot=False) -> None:
        self.env = env
        self.model = model
        self.xarm_ip=xarm_ip
        self._real_control_mode = real_control_mode
        self.robot_action_scale = robot_action_scale
        self.enable_real_robot = enable_real_robot
        if self.enable_real_robot:
            self._configure_real_robot()
    
    def reset(self):
        obs = self.env.reset()
        if self.enable_real_robot:
            self.real_robot.reset()

        return obs
    
    def step(self, exe_action):
        obs, action, done, info = self.env.step(exe_action)
        if self.enable_real_robot:
            self.real_robot.set_action(exe_action, wait=True)

        return obs, action, done, info
    
    def predict(self, obs, deterministic):
        action = self.model.predict(obs, deterministic=deterministic)
        return action

    def _configure_real_robot(self):
        """Create real robot agent"""
        self.real_robot = XArm7(
            self.xarm_ip, control_mode=self._real_control_mode,
            safety_boundary=[550, 0, 50, -600, 280, 0]
        )

def main():
    env_id = 'PickCube-v3'
    log_dir = "./logs/PPO/PickCube-v3"
    record_dir = "logs/PPO/"+env_id
    rollout_steps = 4000
    num_envs = 16
    obs_mode = "state"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "dense"
    low_level_control_mode = 'position'
    motion_data_type = ['qpos', 'qvel', 'qacc', 'qf - passive_qf', 'qf']
    sim_params = generate_sim_params()
    render_mode = 'cameras' # 'human', 'cameras'
    fix_task_configuration = True
    render_by_sim_step = False
    paused = False
    ee_type = 'reduced_gripper' #'reduced_gripper', 'full_gripper'

    enable_real_robot = True

    # import real_robot.envs
    import mani_skill2.envs
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        control_mode=control_mode,
        low_level_control_mode=low_level_control_mode,
        motion_data_type=motion_data_type,
        sim_params=sim_params,
        fix_task_configuration=fix_task_configuration,
        render_by_sim_step=render_by_sim_step,
        paused=paused,
        ee_type=ee_type,
    )

    env = SuccessInfoWrapper(env)
    env = RecordEpisode(env, record_dir, info_on_video=True, render_mode=render_mode, motion_data_type=motion_data_type)

    #-----Load ppo policy-----#
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=rollout_steps // num_envs,
        batch_size=400,
        gamma=0.8,     # default = 0.85
        gae_lambda=0.9,
        n_epochs=20,
        tensorboard_log=log_dir,
        target_kl=0.1,
    )

    model_path = osp.join(log_dir, "best_model")
    # Load the saved model
    model = model.load(model_path)

    #-----Instantiate eval object-----#
    realroboteval = RealRobotEval(env=env, model=model, enable_real_robot=enable_real_robot)

    done = False
    obs = realroboteval.reset()
    num_step = 0
    while not done:
        print(num_step)
        action = realroboteval.predict(obs, deterministic=True)[0]
        obs, action, done, info = realroboteval.step(action)
        num_step += 1

if __name__ == '__main__':
    main()