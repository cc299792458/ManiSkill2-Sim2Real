# Import required packages
import argparse
import os.path as osp

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode, SuccessInfoWrapper, ContinuousTaskWrapper
from mani_skill2.utils.generate_sim_params import generate_sim_params

                    
def parse_args():
    env_id = "PingPong-v0"
    parser = argparse.ArgumentParser(description="Use Stable-Baselines-3 PPO to train ManiSkill2 tasks")
    #####----- PPO Args -----#####
    parser.add_argument("-e", "--env-id", type=str, default=env_id)
    parser.add_argument("--seed", type=int, help="Random seed to initialize training with",)
    parser.add_argument("--max-episode-steps", type=int, default=100, help="Max steps per episode before truncating them")
    #####----- Env Args -----#####
    parser.add_argument("--ee-type", type=str, default='reduced_gripper', help="End effector type") # 'reduced_gripper', 'full_gripper'
    parser.add_argument("--enable-tgs", action="store_true", help="Enable tgs or not")
    parser.add_argument("--obs-noise", type=float, default=0.0, help="Observation noise")
    parser.add_argument("--ee-move-first", type=bool, default=True, help="In one action, finish moving ee first, then move the arm." )
    parser.add_argument("--size-range", type=float, default=0.0, help="Range for object's size." )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    #####----- PPO Args -----#####
    env_id = args.env_id
    #####----- Env Args -----#####
    obs_mode = "state"
    reward_mode = "dense"
    control_mode = "constvel_ee_delta_pose"   # "pd_ee_delta_pose", "constvel_ee_delta_pose"
    low_level_control_mode = 'position'
    motion_data_type = ['qpos', 'qvel', 'qacc', '(qf - passive_qf)', 'qf', 'ee_pos']
    ee_type = args.ee_type 
    enable_tgs = args.enable_tgs
    obs_noise = args.obs_noise
    ee_move_first =  args.ee_move_first
    domain_randomization = args.domain_randomization
    #####----- Debug Args -----#####
    render_mode = 'cameras' # 'human', 'cameras'    
    fix_task_configuration = False
    render_by_sim_step = False
    paused = False
    
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        control_mode=control_mode,
        low_level_control_mode=low_level_control_mode,
        motion_data_type=motion_data_type,
        sim_params = generate_sim_params(),
        ee_type=ee_type,
        enable_tgs=enable_tgs,
        obs_noise=obs_noise,
        ee_move_first=ee_move_first,
        domain_randomization=domain_randomization,
        #####----- Debug Args -----#####
        fix_task_configuration = fix_task_configuration,
        render_by_sim_step = render_by_sim_step,
        paused=paused,
    )

    env.seed(args.seed)
    env.reset()


if __name__ == "__main__":
    main()