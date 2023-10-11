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

from mani_skill2.utils.handcraft_policy import PickCubeV3HandcraftPolicy
                    
def parse_args():
    env_id = "PickCube-v3"
    parser = argparse.ArgumentParser(description="Use Stable-Baselines-3 PPO to train ManiSkill2 tasks")
    #####----- PPO Args -----#####
    parser.add_argument("-e", "--env-id", type=str, default=env_id)
    parser.add_argument("-n", "--n-envs", type=int, default=16, help="Number of parallel envs to run.")
    parser.add_argument("--train", action="store_true", help="Whether to train the policy")
    parser.add_argument("--model-path", type=str, help="Path to sb3 model for evaluation")
    parser.add_argument("--seed", type=int, help="Random seed to initialize training with",)
    parser.add_argument("--max-episode-steps", type=int, default=100, help="Max steps per episode before truncating them")
    parser.add_argument("--total-timesteps", type=int, default=8_000_000, help="Total timesteps for training")
    parser.add_argument("--rollout-steps", type=int, default=4000, help="Rollout steps for PPO." )    # 10000
    parser.add_argument("--batch-size", type=int, default=400, help="Batch size for PPO." )
    parser.add_argument("--gamma", type=float, default=0.8, help="Gamma for PPO." )
    parser.add_argument("--n-epochs", type=int, default=20, help="N epochs for PPO." )
    parser.add_argument("--log-dir", type=str, default="logs/PPO/", help="Path for where logs, checkpoints, and videos are saved")
    parser.add_argument("--pre-trained", action="store_true", help="If using pre-trained model or not")
    parser.add_argument("--pre-trained-dir", type=str, default="ManiSkill2-Sim2Real/train_eval/pre_trained/",help="Dir of pretrained model")
    parser.add_argument("--tensorboard-log-dir", type=str, default="/chichu-slow-vol/tensorboard/", help="Dir of tensorboard log")
    #####----- Env Args -----#####
    parser.add_argument("--ee-type", type=str, default='reduced_gripper', help="End effector type") # 'reduced_gripper', 'full_gripper'
    parser.add_argument("--ee-move-independently", action="store_true", help="One action can only move arm or ee in one time.")
    parser.add_argument("--enable-tgs", action="store_true", help="Enable tgs or not")
    parser.add_argument("--obs-noise", type=float, default=0.0, help="Observation noise")
    parser.add_argument("--ee-move-first", type=bool, default=True, help="In one action, finish moving ee first, then move the arm." )
    parser.add_argument("--size-range", type=float, default=0.0, help="Range for object's size." )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    record_num = 10
    env_id = args.env_id
    log_dir = args.log_dir + env_id
    #####----- Env Args -----#####
    obs_mode = "state"
    reward_mode = "dense"
    control_mode = "constvel_ee_delta_pose"   # "pd_ee_delta_pose", "constvel_ee_delta_pose"
    low_level_control_mode = 'position'
    motion_data_type = ['qpos', 'qvel', 'qacc', '(qf - passive_qf)', 'qf', 'ee_pos']
    ee_type = args.ee_type 
    ee_move_independently = args.ee_move_independently
    enable_tgs = True
    obs_noise = args.obs_noise
    ee_move_first =  args.ee_move_first
    size_range = args.size_range
    #####----- Debug Args -----#####
    render_mode = 'cameras' # 'human', 'cameras'    
    fix_task_configuration = False
    render_by_sim_step = True
    paused = False
    
    if args.seed is not None:
        set_random_seed(args.seed)

    def make_env(env_id: str, max_episode_steps: int = None, record_dir: str = None):
        def _init() -> gym.Env:
            # NOTE: Import envs here so that they are registered with gym in subprocesses
            import mani_skill2.envs

            env = gym.make(
                env_id,
                obs_mode=obs_mode,
                reward_mode=reward_mode,
                control_mode=control_mode,
                low_level_control_mode=low_level_control_mode,
                motion_data_type=motion_data_type,
                sim_params = generate_sim_params(),
                ee_type=ee_type,
                ee_move_independently=ee_move_independently,
                enable_tgs=enable_tgs,
                obs_noise=obs_noise,
                ee_move_first=ee_move_first,
                size_range=size_range,
                #####----- Debug Args -----#####
                fix_task_configuration = fix_task_configuration,
                render_by_sim_step = render_by_sim_step,
                paused=paused,
            )
            # For training, we regard the task as a continuous task with infinite horizon.
            # you can use the ContinuousTaskWrapper here for that
            if max_episode_steps is not None:
                env = ContinuousTaskWrapper(env, max_episode_steps)
            if record_dir is not None:
                env = SuccessInfoWrapper(env)
                env = RecordEpisode(env, record_dir, info_on_video=True, render_mode=render_mode, motion_data_type=motion_data_type)
            
            return env

        return _init

    # create eval environment
    if not args.train:
        record_dir = osp.join(log_dir, "videos/eval")
    else:
        record_dir = osp.join(log_dir, "videos")

    env = make_env(env_id, record_dir=record_dir)()
    env.seed(args.seed)
    policy = PickCubeV3HandcraftPolicy()

    recorded_num = 0
    obs = env.reset()
    
    while recorded_num < record_num:
        action = policy.predict(obs)
        obs, reward, done, info = env.step(action)
        if done == True:
            record_num += 1
            obs = env.reset()
            policy.reset()


if __name__ == "__main__":
    main()