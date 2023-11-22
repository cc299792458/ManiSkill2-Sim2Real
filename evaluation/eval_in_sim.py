import numpy as np

import torch
import torch.nn as nn

import gym
import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
# from mani_skill2.utils.kinematics_helper import PartialKinematicModel, compute_inverse_kinematics

# import time
# from xarm import XArmAPI
from collections import defaultdict
# from sapien.core import Pose
# from transforms3d.euler import euler2quat, quat2euler
# from transforms3d.quaternions import axangle2quat


# import sapien.core as sapien

LOG_STD_MAX = 2
LOG_STD_MIN = -5
SPEED = 6 / 57  # why?

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.action_scale = torch.FloatTensor((env.single_action_space.high - env.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((env.single_action_space.high + env.single_action_space.low) / 2.0)

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

# def collect_episode_info(info, result=None):
#     if result is None:
#         result = defaultdict(list)
#     for item in info:
#         if "episode" in item.keys():
#             print(f"global_step={global_step}, episodic_return={item['episode']['r']:.4f}, success={item['success']}")
#             result['return'].append(item['episode']['r'])
#             result['len'].append(item["episode"]["l"])
#             result['success'].append(item['success'])
#     return result

def make_env(env_id, seed, control_mode=None, video_dir=None):
    def thunk():
        env = gym.make(env_id, reward_mode='dense', obs_mode='state', control_mode=control_mode, render_by_sim_step=True)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True)
            # env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def evaluate(n, agent, eval_envs, device):
    print('======= Evaluation Starts =========')
    result = defaultdict(list)
    obs = eval_envs.reset()
    while len(result['return']) < n:
        with torch.no_grad():
            action = agent.get_eval_action(torch.Tensor(obs).to(device))
        obs, rew, done, info = eval_envs.step(action.cpu().numpy())
        # collect_episode_info(info, result)
    print('======= Evaluation Ends =========')
    return result

if __name__ == '__main__':
    ##### Arguments #####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = "PegInsertionSide2D-v4"
    seed = 0
    control_mode = 'constvel_ee_delta_xy'
    video_dir = "./logs/SAC/PegInsertionSide2D-v4"
    num_eval_envs = 1
    kwargs = {'context': 'forkserver'}
    
    ##### BUild Evaluation Environment #####
    eval_envs = gym.vector.AsyncVectorEnv([make_env(env_id, seed, control_mode, video_dir) for i in range(num_eval_envs)], **kwargs)

    ##### Load actor #####
    ckpt_path = '/home/chichu/Documents/Sapien/ManiSkill2-Sim2Real/evaluation/ckpt/peginsertion2d.pt'
    ckpt = torch.load(ckpt_path)
    agent = Actor(eval_envs).to(device)
    agent.load_state_dict(ckpt['actor'])

    ##### Loop ####    robot.get_obs()
    obs = eval_envs.reset()
    while True:
        with torch.no_grad():
            action = agent.get_eval_action(torch.Tensor(obs).to(device)).cpu().numpy()
            obs, rew, done, info = eval_envs.step(action)