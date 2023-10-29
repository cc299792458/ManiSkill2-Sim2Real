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
    env_id = "PickCube-v3"
    parser = argparse.ArgumentParser(description="Use Stable-Baselines-3 PPO to train ManiSkill2 tasks")
    #####----- PPO Args -----#####
    parser.add_argument("-e", "--env-id", type=str, default=env_id)
    parser.add_argument("-n", "--n-envs", type=int, default=16, help="Number of parallel envs to run.")
    parser.add_argument("--train", action="store_true", help="Whether to train the policy")
    parser.add_argument("--model-path", type=str, help="Path to sb3 model for evaluation")
    parser.add_argument("--seed", type=int, help="Random seed to initialize training with")
    parser.add_argument("--max-episode-steps", type=int, default=50, help="Max steps per episode before truncating them")
    parser.add_argument("--total-timesteps", type=int, default=8_000_000, help="Total timesteps for training")
    parser.add_argument("--rollout-steps", type=int, default=4000, help="Rollout steps for PPO." )    # 10000
    parser.add_argument("--batch-size", type=int, default=400, help="Batch size for PPO." )
    parser.add_argument("--gamma", type=float, default=0.8, help="Gamma for PPO." )
    parser.add_argument("--n-epochs", type=int, default=20, help="N epochs for PPO." )
    parser.add_argument("--log-dir", type=str, default="logs/PPO/", help="Path for where logs, checkpoints, and videos are saved")
    parser.add_argument("--pre-trained", action="store_true", help="If using pre-trained model or not")
    parser.add_argument("--pre-trained-dir", type=str, default="ManiSkill2-Sim2Real/train_eval/pre_trained/",help="Dir of pretrained model")
    parser.add_argument("--tensorboard-log-dir", type=str, default="logs/PPO/", help="Dir of tensorboard log")
    # parser.add_argument("--tensorboard-log-dir", type=str, default="/chichu-slow-vol/tensorboard/", help="Dir of tensorboard log")
    #####----- Env Args -----#####
    parser.add_argument("--ee-type", type=str, default='reduced_gripper', help="End effector type") # 'reduced_gripper', 'full_gripper'
    parser.add_argument("--enable-tgs", action="store_true", help="Enable tgs or not")
    parser.add_argument("--ee-move-first", type=bool, default=True, help="In one action, finish moving ee first, then move the arm." )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    #####----- PPO Args -----#####
    env_id = args.env_id
    num_envs = args.n_envs
    max_episode_steps = args.max_episode_steps
    rollout_steps = args.rollout_steps
    batch_size = args.batch_size
    gamma = args.gamma
    n_epochs = args.n_epochs
    log_dir = args.log_dir + env_id
    pre_trained_dir = args.pre_trained_dir + env_id
    tensorboard_log_dir = args.tensorboard_log_dir + env_id
    #####----- Env Args -----#####
    obs_mode = "state"
    reward_mode = "dense"
    control_mode = "pd_ee_delta_pose"   # "pd_ee_delta_pose", "constvel_ee_delta_pose"
    low_level_control_mode = 'impedance' # position, impedance
    motion_data_type = ['qpos', 'qvel', 'qacc', '(qf - passive_qf)', 'qf', 'ee_pos']
    ee_type = args.ee_type 
    enable_tgs = args.enable_tgs
    ee_move_first =  args.ee_move_first
    domain_rand_params = None
    # domain_rand_params = dict(size_range=0.005, fric_range=[0.5, 1.5], obs_noise=0.005)
    #####----- Debug Args -----#####
    render_mode = 'cameras' # 'human', 'cameras'    
    fix_task_configuration = True
    render_by_sim_step = False
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
                enable_tgs=enable_tgs,
                ee_move_first=ee_move_first,
                #####----- Debug Args -----#####
                fix_task_configuration = fix_task_configuration,
                render_by_sim_step = render_by_sim_step,
                paused=paused,
                domain_rand_params=domain_rand_params,
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
    eval_env = SubprocVecEnv([make_env(env_id, record_dir=record_dir) for _ in range(1)])
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(args.seed)
    eval_env.reset()

    if not args.train:
        env = eval_env
    else:
        # Create vectorized environments for training
        env = SubprocVecEnv([make_env(env_id, max_episode_steps=max_episode_steps) for _ in range(num_envs)])
        env = VecMonitor(env)
        env.seed(args.seed)
        env.reset()

    # Define the policy configuration and algorithm configuration
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        n_steps=rollout_steps // num_envs,  # 10000
        batch_size=batch_size,
        gamma=gamma,     # default = 0.85
        gae_lambda=0.9,
        n_epochs=n_epochs,    # 5
        tensorboard_log=tensorboard_log_dir,
        target_kl=0.1,  
        verbose=1,
    )

    if not args.train:
        model_path = args.model_path
        if model_path is None:
            model_path = osp.join(log_dir, "best_model")
        # Load the saved model
        model = model.load(model_path)
    else:
        # define callbacks to periodically save our model and evaluate it to help monitor training
        # the below freq values will save every 20 rollouts
        if args.pre_trained:
            model_path = args.model_path
            if model_path is None:
                model_path = osp.join(pre_trained_dir, "pre_trained_model")
            model = model.load(model_path)
            model.set_env(env)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=20 * rollout_steps // num_envs,
            deterministic=True,
            render=False,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=20 * rollout_steps // num_envs,
            save_path=log_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        # Train an agent with PPO for args.total_timesteps interactions
        model.learn(
            args.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
        # Save the final model
        model.save(osp.join(log_dir, "best_model"))

    # Evaluate the model
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=True,
        return_episode_rewards=True,
        n_eval_episodes=10,
    )
    print("Returns", returns)
    print("Episode Lengths", ep_lens)
    success = np.array(ep_lens) < 50
    success_rate = success.mean()
    print("Success Rate:", success_rate)


if __name__ == "__main__":
    main()