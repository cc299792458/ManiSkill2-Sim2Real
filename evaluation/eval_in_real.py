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


import cv2
import pyrealsense2 as rs
import imageio
import copy

def  check_color(img,color):
    if color == "R":
        if img[0]>100:
            if img[1]<60:
                if img[2]<60:
                    return True
        if img[0]>120:
            if img[1]<80:
                if img[2]<80:
                    return True
    return False
# def not_chess(color):
#     if color[0]<120 and color[1]<120 and color[2]<120:
#         return False
#     if 180<color[0] and 180<color[1] and 180<color[2]:
#         return False
#     return True

def is_chess(color):
    threshold = 45
    if np.abs(int(color[0])-int(color[1])) < threshold and np.abs(int(color[2])-int(color[1])) < threshold and np.abs(int(color[0])-int(color[2])) < threshold:
        return True
    return False
    
def get_real_XY(output_point_1,output_point_2):
    theta = np.arctan(np.abs(output_point_2[1] - output_point_1[1]) / np.abs(output_point_2[0] - output_point_1[0]))
    # beta = 90 - theta
    # beta = np.arctan(np.abs(output_point_2[0] - output_point_1[0]) / np.abs(output_point_2[1] - output_point_1[1]))
    length = 0.15 / 0.05 * 80#(np.abs(output_point_2[1] - output_point_1[1])**2 + np.abs(output_point_2[0] - output_point_1[0])**2)**0.5
    width = length / 3
    mid_point = (output_point_1 + output_point_2) / 2
    delta_y = width / 2 * np.cos(theta)
    delta_x = width / 2 * np.sin(theta)
    x = mid_point[0] - delta_x
    y = mid_point[1] - delta_y
    # print((720-x)/80*0.05,y/80*0.05)
    real_x = (x - 560) / 80 * 0.05 + (0.4669 - 0.464) 
    real_y = -(y) / 80 * 0.05 - 0.0681
    
    return real_x, real_y, theta

def get_pixel(color_image):
    stop=False
    for i in range(300):
        i = 299 - i
        if stop ==True: break
        for j in range(450):
            j = 449 - j
            if stop ==True: break
            if check_color(color_image[i,j],"R"):
                x1=i
                y1=j
                stop = True
                # print("corner 1", x1, y1)
                # print("RGB 1", color_image[i,j])     
    if stop == True:
        for i in range(x1-20,x1+20):
            for j in range(y1-20,y1+20):
                color_image[i,j]=(0,0,0)
    stop = False
    for i in range(300):
        if stop ==True: break
        i = 299 - i
        for j in range(450):
            if stop ==True: break
            j = 449 - j
            if stop ==True: break
            if check_color(color_image[i,j],"R"):
                x2=i
                y2=j
                stop = True
                # print("corner 2", x2, y2)
                # print("RGB 2", color_image[i,j])

    if stop == False:
        return False, 0, 0, 0, 0
    return stop, x1, y1, x2, y2

def get_mean_pixel(color_image):
    stop=False
    for j in range(450):
        if stop==True: break
        for i in range(300):
            if stop ==True: break
            if check_color(color_image[i,j],"R"):
                x1=i
                y1=j
                stop = True
                color_image[i,j] = (0,0,0)
                # print("corner 1", x1, y1)
                # print("RGB 1", color_image[i,j])     
    if stop == True:
        total1 = 1
        for i in range(x1-20,x1+20):
            for j in range(y1-20,y1+20):
                if check_color(color_image[i,j],"R"):
                    x1+=i
                    y1+=j
                    color_image[i,j] = (0,0,0)
                    total1 += 1
        x1 = x1 / total1
        y1 = y1 / total1

    stop = False
    for j in range(450):
        j = 449 - j
        if stop==True: break
        for i in range(300):
            i = 299 - i
            if stop ==True: break
            if check_color(color_image[i,j],"R"):
                x2=i
                y2=j
                stop = True
                # print("corner 2", x2, y2)
                # print("RGB 2", color_image[i,j])
    if stop == True:
        total2 = 1
        for i in range(x2-10,x2+10):
            for j in range(y2-10,y2+10):
                # print(i,j)
                if check_color(color_image[i,j],"R"):
                    x2+=i
                    y2+=j
                    color_image[i,j] = (0,0,0)
                    total2 += 1
        x2 = x2 / total2
        y2 = y2 / total2

    if stop == False:
        return False, 0, 0, 0, 0
    return stop, int(x1), int(y1), int(x2), int(y2)

def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)  # RGB流
    profile = pipeline.start(config)
    profile.get_device().sensors[1].set_option(rs.option.exposure, 86)
    for _ in range(150):    
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

    return pipeline

# pipeline = initialize_realsense()
def get_pose():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)  # RGB流
    profile = pipeline.start(config)
    profile.get_device().sensors[1].set_option(rs.option.exposure, 86)
    for _ in range(150):    
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
    ret = False
    while ret == False:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        img = copy.deepcopy(color_image)
        color_image=color_image[225:525,450:900]
        imageio.imsave("before.png",color_image)
        stop = False
        ret, x1, y1, x2, y2=get_pixel(color_image) 

    H = np.load('/home/chichu/Downloads/H.npy')

    #np.array and image are transposes
    input_point_1 = np.array([y1+450,x1+225, 1])
    output_point_1 = np.matmul(H, input_point_1)
    output_point_1[0] /= output_point_1[2]
    output_point_1[1] /= output_point_1[2]

    #temp = output_point_1[0]
    #output_point_1[0] = output_point_1[1]
    #output_point_1[1] = temp

    input_point_2 = np.array([y2+450,x2+225, 1])
    output_point_2 = np.matmul(H, input_point_2)
    output_point_2[0] /= output_point_2[2]
    output_point_2[1] /= output_point_2[2]

    #temp = output_point_2[0]
    #output_point_2[0] = output_point_2[1]
    #output_point_2[1] = temp

    #print("output_point_1:", output_point_1[0:2])
    #print("output_point_2:", output_point_2[0:2])

    return get_real_XY(output_point_1,output_point_2)

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
        if self.env_name == 'PickCube':
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
        elif self.mode == 'impedance':
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
        elif self.mode == 'impedance':
            self.arm.vc_set_joint_velocity(self._preprocess_arm_action(action[0:3]), is_radian=True, is_sync=True, duration=self.duration)
            self.arm.set_gripper_position(self._preprocess_gripper_action(action[3]), is_sync=False, speed=SPEED, wait=False)
    
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

        elif self.mode == 'impedance':
            action = np.hstack([arm_action, np.zeros([3])])
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
        else:
            tcp_pose = vectorize_pose(self.tcp_pose)
            peg_x, peg_y, theta = get_pose()
            peg_q = euler2quat(0.0, 0.0, theta + np.pi)
            print(peg_x, peg_y)
            peg_p = np.array([peg_x, peg_y, 0.025])
            peg_pose = np.hstack([peg_p, peg_q])
            box_hole_pose = np.array([-0.4, -0.2, 0.025, 0.0, 0.0, 0.0, 1.0])

            obs = np.hstack([self.qpos, self.qvel, tcp_pose, peg_pose, box_hole_pose, self.obj_grasped])
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
        return np.array([-0.35, -0.25, 0.02, 1.0, 0.0, 0.0, 0.0])

    @property
    def obj_grasped(self):
        _, gripper_dis = self.arm.get_gripper_position()
        return float(gripper_dis < 475)

if __name__ == '__main__':
    # print(get_pose())
    ##### Arguments #####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = "PegInsertionSide2D-v4"
    seed = 0
    control_mode = 'constvel_ee_delta_xy'
    video_dir = None
    num_eval_envs = 1
    kwargs = {'context': 'forkserver'}
    
    ##### BUild Evaluation Environment #####
    eval_envs = gym.vector.AsyncVectorEnv([make_env(env_id, seed, control_mode, video_dir) for i in range(num_eval_envs)], **kwargs)

    ##### Load actor #####
    ckpt_path = '/home/chichu/Documents/Sapien/ManiSkill2-Sim2Real/evaluation/ckpt/peginsertion2d.pt'
    ckpt = torch.load(ckpt_path)
    agent = Actor(eval_envs).to(device)
    agent.load_state_dict(ckpt['actor'])

    ##### Instantiate realrobot #####
    virtual_envs = gym.make(env_id, reward_mode='dense', obs_mode='state', control_mode=control_mode)
    robot = RealXarm(env=virtual_envs, ip="192.168.1.212", mode='position_xy', env_name='peginsertion2d')
    # robot.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    ##### Loop ####
    
    obs_sim = eval_envs.reset()
    flag = False
    while True:
        with torch.no_grad():
            obs = robot.get_obs()
            # if flag == False:
            #     obs[20], obs[21] = -obs[20], -obs[21]
            #     flag = True
            # obs[20], obs[21] = -obs[20], -obs[21]
            # obs_sim = eval_envs.reset()
            # diff = np.abs(obs - obs_sim)
            action = agent.get_eval_action(torch.Tensor(obs).to(device)).cpu().numpy()
            # obs_sim, rew, done, info = eval_envs.step(action[np.newaxis, :])
            robot.step(action)
            # obs_sim, rew, done, info = eval_envs.step(action)
