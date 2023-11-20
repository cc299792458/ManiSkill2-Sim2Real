import numpy as np

import torch
import torch.nn as nn

import gym
import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.utils.kinematics_helper import PartialKinematicModel, compute_inverse_kinematics
from mani_skill2.utils.sapien_utils import vectorize_pose

import time
from xarm import XArmAPI
from collections import defaultdict
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import axangle2quat


import sapien.core as sapien

LOG_STD_MAX = 2
LOG_STD_MIN = -5
SPEED = 12 / 57  # why?

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

def make_env(env_id, seed, control_mode=None, video_dir=None):
    def thunk():
        env = gym.make(env_id, reward_mode='dense', obs_mode='state', control_mode=control_mode)
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

class RealXarm:
    def __init__(self, env, ip, control_freq=20, mode='position_pos', env_name='PickCube'):
        self.env = env
        self.ip = ip
        self.control_freq = control_freq
        self.duration = 1 / control_freq
        self.mode = mode
        self.env_name = env_name
        self._init_arm()

    def _init_arm(self):
        self.arm = XArmAPI(self.ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_gripper_enable(True)
        if self.env_name == 'PickCube' or 'graspcube':
            qpos = np.array([0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2])
            self.arm.set_servo_angle(angle=qpos[:7], is_radian=True, wait=True, speed=SPEED)
        if self.env_name == 'peginsertion2d':
            qpos = np.array([0.0, 0.0, 0.0, np.pi / 6, 0.0, np.pi / 6, 0.0])
            self.arm.set_servo_angle(angle=qpos[:7], is_radian=True, wait=True, speed=SPEED)
        self.arm.set_gripper_position(850, wait=True, speed=SPEED)
        time.sleep(1)
        if self.mode =='position_pos':
            self.translation_scale = 100
        elif self.mode == 'position_pose':
            self.translation_scale = 100
            self.axangle_scale = 0.1
        elif 'impedance' in self.mode:
            self.translation_scale = 0.05
            self.cur_time = self.start_time = time.time()
            self.arm.set_mode(4)
            self.arm.set_state(state=0)
            self.arm.set_gripper_enable(True)
            self._load_virtual_robot()
            self._init_vel_ik()

    def _load_virtual_robot(self, robot_name='xarm') -> sapien.Articulation:
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / 500.0)
        loader = self.scene.create_urdf_loader()

        filename = "/home/chichu/Documents/Sapien/ManiSkill2-Sim2Real/mani_skill2/assets/descriptions/xarm7_reduced_gripper_d435.urdf"
        robot_builder = loader.load_file_as_articulation_builder(filename)
        self.articulation = robot_builder.build(fix_root_link=True)
        self.articulation.set_name(robot_name)
    
    def _init_vel_ik(self):
        self.start_joint_name = self.articulation.get_joints()[1].get_name()
        self.end_joint_name = self.articulation.get_active_joints()[6].get_name()
        self.kinematic_model = PartialKinematicModel(self.articulation, self.start_joint_name, self.end_joint_name)
    
    def step(self, action):
        if self.mode =='position_pos':
            self.arm.set_gripper_position(self._preprocess_gripper_action(action[3]), wait=True)
            self.env.agent.robot.set_qpos(self.qpos)
            # tcp_at_base = self.tcp_pose[0:3]
            # tcp_at_base[0] += 0.4638637
            tcp_pose_at_base = self.tcp_pose_at_base
            cur_tcp_pose = Pose(p=tcp_pose_at_base.p, q=tcp_pose_at_base.q)
            delta_tcp_pose = Pose(p=action[0:3] * 0.1, q=[1.0, 0.0, 0.0, 0.0])
            target_tcp_pose = cur_tcp_pose * delta_tcp_pose
            target_qpos = self.env.agent.controller.controllers['arm'].compute_ik(target_tcp_pose)
            self.arm.set_servo_angle(angle=target_qpos, is_radian=True, wait=True)
            
            # delta_tcp_pose = self._preprocess_arm_action(action[0:3])
            # ret_arm = self.arm.set_tool_position(
            #     *delta_tcp_pose.p, *quat2euler(delta_tcp_pose.q, axes='sxyz'),
            #     is_radian=True, wait=True)
        elif self.mode == 'position_pose':
            self.arm.set_gripper_position(self._preprocess_gripper_action(action[6]), wait=True)
            delta_tcp_pose = self._preprocess_arm_action(action[0:6])
            ret_arm = self.arm.set_tool_position(
                *delta_tcp_pose.p, *quat2euler(delta_tcp_pose.q, axes='sxyz'),
                is_radian=True, wait=True)
        elif self.mode == 'position_xy':
            self.arm.set_gripper_position(self._preprocess_gripper_action(action[2]), wait=True)
            self.env.agent.robot.set_qpos(self.qpos)
            # tcp_at_base = self.tcp_pose[0:3]
            # tcp_at_base[0] += 0.4638637
            tcp_pose_at_base = self.tcp_pose_at_base
            cur_tcp_pose = Pose(p=tcp_pose_at_base.p, q=tcp_pose_at_base.q)
            delta_tcp_pose = Pose(p=np.hstack([action[0:2], np.zeros([1])]) * 0.1, q=[1.0, 0.0, 0.0, 0.0])
            target_tcp_pose = cur_tcp_pose * delta_tcp_pose
            target_qpos = self.env.agent.controller.controllers['arm'].compute_ik(target_tcp_pose)
            self.arm.set_servo_angle(angle=target_qpos, is_radian=True, wait=True)
        elif self.mode == 'impedance_pos':
            self.cur_time = time.time()
            while self.cur_time - self.start_time < 0.05:
                time.sleep(0.001)
                self.cur_time = time.time()
            self.arm.vc_set_joint_velocity(self._preprocess_arm_action(action[0:3]), is_radian=True, is_sync=True, duration=self.duration)
            self.arm.set_gripper_position(self._preprocess_gripper_action(action[3]), is_sync=False, speed=SPEED, wait=False)
            self.start_time = time.time()
    
    def _preprocess_arm_action(self, arm_action):
        if self.mode =='position_pos':
            delta_tcp_pose = Pose(p=arm_action * self.translation_scale,  # in milimeters
                                  q=np.array([1.0, 0.0, 0.0, 0.0]))
            return delta_tcp_pose
        elif self.mode == 'position_pose':
            cur_tcp_pose = Pose(p=self.tcp_pose[0:3] * self.translation_scale, q=self.tcp_pose[3:])
            axangle = arm_action[3:6]
            rot_angle = np.linalg.norm(axangle)
            delta_tcp_pose = Pose(p=arm_action[:3] * self.translation_scale,  # in milimeters
                                  q=axangle2quat(axangle / (rot_angle + 1e-9),
                                                 np.clip(rot_angle, 0.0, 1.0) * self.axangle_scale))
            return delta_tcp_pose

        elif self.mode == 'impedance_pos':
            action = np.hstack([arm_action * self.translation_scale, np.zeros([3])])
            palm_jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(self.qpos[:-2])
            target_qvel = np.clip(compute_inverse_kinematics(action, palm_jacobian), -np.pi/2, np.pi/2)

            return target_qvel

    def _preprocess_gripper_action(self, gripper_action):
        action = (850 + (-10)) / 2 * (1 + gripper_action)
        return action

    def get_obs(self):
        if self.env_name == 'PickCube':
            tcp_pose = vectorize_pose(self.tcp_pose)        # tcp pose should be calculated by virtual env

            goal_pos = self.goal_pos
            tcp_to_goal_pos = goal_pos - tcp_pose[0:3]
            obj_pose = self.obj_pose
            tcp_to_obj_pos = obj_pose[0:3] - tcp_pose[0:3]
            obj_to_goal_pos = goal_pos - obj_pose[0:3]
            obs = np.hstack([self.qpos, self.qvel, tcp_pose, goal_pos, tcp_to_goal_pos,
                            obj_pose, tcp_to_obj_pos, obj_to_goal_pos, self.obj_grasped])
            return obs
        elif self.env_name == 'peginsertion2d':
            tcp_pose = vectorize_pose(self.tcp_pose)
            peg_x = float(input('x:'))
            peg_y = float(input('y:'))
            peg_pose = np.array([peg_x, peg_y, 0.025, 0.0, 0.0, 0.0, 1.0])
            box_hole_pose = np.array([-0.4, -0.2, 0.025, 0.0, 0.0, 0.0, 1.0])

            obs = np.hstack([self.qpos, self.qvel, tcp_pose, peg_pose, box_hole_pose, self.obj_grasped])
            
            return obs
        
        elif self.env_name == 'graspcube':
            tcp_pose = vectorize_pose(self.tcp_pose)        # tcp pose should be calculated by virtual env
            obj_pose = self.obj_pose
            tcp_to_obj_pos = obj_pose[0:3] - tcp_pose[0:3]
            obs = np.hstack([self.qpos, self.qvel, tcp_pose, obj_pose, tcp_to_obj_pos])
            
            return obs
        
    def gripper_real_2_sim(self, gripper_dis):
        return gripper_dis * 5.249186 * 1e-5 + 2.49186 * 1e-5

    @property
    def qpos(self):
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        _, gripper_dis = self.arm.get_gripper_position()
        gripper_qpos = self.gripper_real_2_sim(gripper_dis)

        return np.hstack([qpos, [gripper_qpos, gripper_qpos]])
        
    @property
    def qvel(self):
        """Get xarm qvel in maniskill2 format"""
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        return np.hstack([qvel])  # No gripper qvel
    
    @property
    def tcp_pose(self):
        """Use the fk of simulation to calculate
        """
        self.env.agent.robot.set_qpos(self.qpos)
        tcp_pose = self.env.agent.controller.controllers['arm'].ee_pose

        return tcp_pose
        # _, base_to_tcp = self.arm.get_forward_kinematics(
        #     self.qpos, input_is_radian=True, return_is_radian=True
        # )
        # base_to_tcp = np.asarray(base_to_tcp)
        # base_to_tcp_pose = np.hstack([base_to_tcp[:3] / 1000, euler2quat(*base_to_tcp[3:], axes='sxyz')])
        
        # tcp_pose = base_to_tcp_pose
        # tcp_pose[0] -= 0.4638637

        # tcp_pose = Pose(p=tcp_pose[0:3], q=tcp_pose[3:7])
        
        # return tcp_pose

    @property
    def tcp_pose_at_base(self):
        self.env.agent.robot.set_qpos(self.qpos)
        tcp_pose_at_base = self.env.agent.controller.controllers['arm'].ee_pose_at_base

        return tcp_pose_at_base

    @property
    def goal_pos(self):
        return np.array([0.0, 0.0, 0.35])

    @property
    def obj_pose(self):
        # [0.0, 0.0], [0.02, 0.02], [0.035, -0.02], [-0.02, -0.04], 
        # *[0.01, -0.045]*
        # [0.0, 0.0, 45], [0.02, 0.02, 45], [-0.035, -0.02, 30], [0.035, -0.03]
        # return np.array([0.035, -0.03, 0.02, 0.9659258, 0.0, 0.0, 0.258819])
        # return np.array([0.05510862, 0.05108298, 0.02, 0.9238795, 0.0, 0.0, 0.3826834])
        return np.array([0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0])

    @property
    def obj_grasped(self):
        _, gripper_dis = self.arm.get_gripper_position()
        return float(gripper_dis < 475)

    ##### Camera related #####

if __name__ == '__main__':
    ##### Arguments #####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = "GraspCube-v1"
    seed = 0
    control_mode = 'pd_ee_vel_pos'
    video_dir = None
    num_eval_envs = 1
    kwargs = {'context': 'forkserver'}
    
    ##### BUild Evaluation Environment #####
    eval_envs = gym.vector.AsyncVectorEnv([make_env(env_id, seed, control_mode, video_dir) for i in range(num_eval_envs)], **kwargs)

    ##### Load actor #####
    ckpt_path = '/home/chichu/Documents/Sapien/ManiSkill2-Sim2Real/evaluation/ckpt/graspcube.pt'
    ckpt = torch.load(ckpt_path)
    agent = Actor(eval_envs).to(device)
    agent.load_state_dict(ckpt['actor'])

    ##### Instantiate realrobot #####
    virtual_envs = gym.make(env_id, reward_mode='dense', obs_mode='state', control_mode=control_mode)
    robot = RealXarm(env=virtual_envs, ip="192.168.1.212", mode='impedance_pos', env_name='graspcube')
    # robot.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    ##### Loop ####
    
    obs_sim = eval_envs.reset()
    flag = False
    while True:
        with torch.no_grad():
            obs = robot.get_obs()
            action = agent.get_eval_action(torch.Tensor(obs).to(device)).cpu().numpy()
            # obs_sim, rew, done, info = eval_envs.step(action[np.newaxis, :])
            robot.step(action)
            if vectorize_pose(robot.tcp_pose)[2] <= 0.02:
                break
            # obs_sim, rew, done, info = eval_envs.step(action)