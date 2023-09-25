from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose

from .base_env import StationaryManipulationEnv


@register_env("PickCube-v0", max_episode_steps=200)
class PickCubeEnv(StationaryManipulationEnv):
    goal_thresh = 0.025
    min_goal_dist = 0.05

    def __init__(self, *args, obj_init_rot_z=True, **kwargs):
        self.obj_init_rot_z = obj_init_rot_z
        self.cube_half_size = np.array([0.019] * 3, np.float32)  # (chichu) change the half size of cube from 0.02 to 0.049/2 to align the real cube.
        # TODO(chichu): need to be removed
        # import sys
        # sys.path.append('/home/chichu/Documents/Sapien/ManiSkill2-Sim2Real')
        # from real_robot.envs.base_env import XArmBaseEnv
        # if not isinstance(self, XArmBaseEnv):
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.obj = self._build_cube(self.cube_half_size)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_actors(self):
        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        xyz = np.hstack([xy, self.cube_half_size[2]])
        q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)
        # NOTE(chichu): fixed to a certain pose when evaluate on real robot with simulation.
        if self.fix_task_configuration:
            xyz = np.array([0.0, 0.0, self.cube_half_size[2]])
            q = [1, 0, 0, 0]
        self.obj.set_pose(Pose(xyz, q))

    def _initialize_task(self, max_trials=100, verbose=False):
        obj_pos = self.obj.pose.p
        # Sample a goal position far enough from the object
        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            goal_z = self._episode_rng.uniform(0, 0.5) + obj_pos[2]
            goal_pos = np.hstack([goal_xy, goal_z])
            # NOTE(chichu): set to a fixed point when evaluate real robot with simulation
            if self.fix_task_configuration:
                goal_pos = np.array([0.0, 0.0, 0.2])
            if np.linalg.norm(goal_pos - obj_pos) > self.min_goal_dist:
                if verbose:
                    print(f"Found a valid goal at {i}-th trial")
                break

        self.goal_pos = goal_pos
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            goal_pos=self.goal_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
                obj_to_goal_pos=self.goal_pos - self.obj.pose.p,
                # Add if the cube is grasped
                obj_grasped=float(self.agent.check_grasp(self.obj)),
            )
        return obs

    def check_obj_placed(self):
        return np.linalg.norm(self.goal_pos - self.obj.pose.p) <= self.goal_thresh

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def evaluate(self, **kwargs):
        is_obj_placed = self.check_obj_placed()
        is_robot_static = self.check_robot_static()
        return dict(
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed and is_robot_static,
        )

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward

        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward

        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        reward += 1 if is_grasped else 0.0

        if is_grasped:
            obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
            place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
            reward += place_reward

        return reward

    def render(self, mode="human"):
        # NOTE(chichu) It seems a bug appears here. Mode would be automatically set to rgb_array if the parameter is 'human',
        # but it won't if the parameter is 'cameras'.
        if mode in ["human", "rgb_array"]:
            self.goal_site.unhide_visual()
            ret = super().render(mode=mode)
            self.goal_site.hide_visual()
        else:
            ret = super().render(mode=mode)
        return ret

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])

# Note: 50 steps is more suitable for position control
@register_env("PickCube-v1", max_episode_steps=50)
class PickCubeEnv_v1(PickCubeEnv):
    # better reward
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward

        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward

        is_grasped = self.agent.check_grasp(self.obj) # remove max_angle=30 yeilds much better performance
        reward += 1 if is_grasped else 0.0

        if is_grasped:
            obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
            place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
            reward += place_reward

            # static reward
            if self.check_obj_placed():
                # hard-coded with xarm full-gripper
                qvel = self.agent.robot.get_qvel()[:-6]
                static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
                reward += static_reward

        return reward


# Note: 50 steps is more suitable for position control
@register_env("PickCube-v2", max_episode_steps=50)
class PickCubeEnv_v2(PickCubeEnv):
    # better reward
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward
        
        if info["time_out"]:
            reward -= 1

        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward

        is_grasped = self.agent.check_grasp(self.obj) # remove max_angle=30 yeilds much better performance
        reward += 1 if is_grasped else 0.0

        if is_grasped:
            obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
            place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
            reward += place_reward

            # static reward
            if self.check_obj_placed():
                qvel = self.agent.robot.get_qvel()[:-2]
                static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
                reward += static_reward

        return reward
    
# Note: 50 steps is more suitable for position control
@register_env("PickCube-v3", max_episode_steps=50)
class PickCubeEnv_v3(PickCubeEnv):
    # better reward
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward
        
        if info["time_out"]:
            reward -= 3

        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward
        # reached = True if tcp_to_obj_dist < 0.05 else False
        
        is_grasped = self.agent.check_grasp(self.obj) # remove max_angle=30 yeilds much better performance
        # if reached:
        if is_grasped:
            reward += 1
            # else:
            #     if self.ee_type == 'reduced_gripper':
            #         gripper_limit = self.agent.robot.get_qlimits()[-1, 1]
            #         reward += 1 - self.agent.robot.get_qpos()[-1] / gripper_limit
            #     elif self.ee_type == 'full_gripper':
            #         gripper_width = self.agent.robot.get_qlimits()[-1, 1] + 0.01
            #         reward += self.agent.robot.get_qpos()[-1] / gripper_width

        if is_grasped:
            obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
            place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
            reward += place_reward

            # static reward
            if self.check_obj_placed():
                if self.ee_type == 'reduced_gripper':
                    qvel = self.agent.robot.get_qvel()[:-2]
                elif self.ee_type == 'full_gripper':
                    qvel = self.agent.robot.get_qvel()[:-6]
                static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
                reward += static_reward

        return reward


@register_env("LiftCube-v0", max_episode_steps=200)
class LiftCubeEnv(PickCubeEnv):
    """Lift the cube to a certain height."""

    goal_height = 0.2

    def _initialize_task(self):
        self.goal_pos = self.obj.pose.p + [0, 0, self.goal_height]
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
            )
        return obs

    def check_obj_placed(self):
        return self.obj.pose.p[2] >= self.goal_height + self.cube_half_size[2]

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 2.25
            return reward

        # reaching reward
        gripper_pos = self.tcp.get_pose().p
        obj_pos = self.obj.get_pose().p
        dist = np.linalg.norm(gripper_pos - obj_pos)
        reaching_reward = 1 - np.tanh(5 * dist)
        reward += reaching_reward

        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)

        # grasp reward
        if is_grasped:
            reward += 0.25

        # lifting reward
        if is_grasped:
            lifting_reward = self.obj.pose.p[2] - self.cube_half_size[2]
            lifting_reward = min(lifting_reward / self.goal_height, 1.0)
            reward += lifting_reward

        return reward
