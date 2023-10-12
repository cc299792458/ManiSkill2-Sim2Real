from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose

from .base_env import StationaryManipulationEnv

@register_env("PingPong-v0", max_episode_steps=200)
class PingPongEnv(StationaryManipulationEnv):
    def __init__(self, *args, ball_radius=0.02, size_range=0.0, **kwargs):
        self.size_range = size_range
        self.ball_radius = ball_radius
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.obj = self._build_ball(self.ball_radius)

    def _initialize_actors(self):
        # if self.size_range != 0.0:
        #     self._actors.remove(self.obj)
        #     self._scene.remove_actor(self.obj)
        #     random_size = self._episode_rng.uniform(0, self.size_range)
        #     half_cube_size = self.org_half_cube_size + random_size
        #     self.cube_half_size = np.array([half_cube_size] * 3, np.float32)
        #     self.cube_half_size[2] = self.org_half_cube_size
        #     self.obj = self._build_cube(self.cube_half_size)
        #     self._actors.append(self.obj)

        xy = self._episode_rng.uniform(-0.005, 0.005, [2])
        z = self._episode_rng.uniform(0.35, 0.45, [1])
        xyz = np.hstack([xy, z])
        if self.fix_task_configuration:
            xyz = np.array([0.0, 0.0, 0.40])
        self.obj.set_pose(Pose(xyz))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pos=self.tcp.pose.p
        )
        if self._obs_mode in ["state", "state_dict"]:
            obj_pos = self.obj.pose.p
            tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
            if self.obs_noise != 0.0:
                xy_noise = self.generate_noise_for_pos(size=2)
                obj_pos[0:2] += xy_noise
                tcp_to_obj_pos[0:2] += xy_noise
            obs.update(
                obj_pose=obj_pos,
                tcp_to_obj_pos=tcp_to_obj_pos,
            )
        return obs

    def generate_noise_for_pos(self, size):
        noise = np.random.uniform(-self.obs_noise, self.obs_noise, size=size)
        return noise

    def check_obj_placed(self):
        return np.linalg.norm(self.goal_pos - self.obj.pose.p) <= self.goal_thresh

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh
    
    def check_reached(self):
        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reached = True if tcp_to_obj_dist < 0.005 else False

        return reached

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

        obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
        if is_grasped:
            if obj_to_goal_dist < self.last_obj_to_goal_dist:
                place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
                reward += place_reward
        self.last_obj_to_goal_dist = obj_to_goal_dist
        

        return reward

    def render(self, mode="human"):
        # NOTE(chichu) It seems a bug appears here. Mode would be automatically set to rgb_array if the parameter is 'human',
        # but it won't if the parameter is 'cameras'.
        ret = super().render(mode=mode)
        return ret

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])
