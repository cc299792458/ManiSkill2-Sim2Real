from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.common import compute_angle_between
from mani_skill2.utils.sapien_utils import (
    vectorize_pose,
    get_entity_by_name,
    get_pairwise_contact_impulse,
)

from .base_env import StationaryManipulationEnv


@register_env("PickCube-v0", max_episode_steps=200)
class PickCubeEnv(StationaryManipulationEnv):
    goal_thresh = 0.025
    min_goal_dist = 0.05

    def __init__(self, *args, obj_init_rot_z=True, **kwargs):
        self.obj_init_rot_z = obj_init_rot_z
        self.cube_half_size = np.array([0.02] * 3, np.float32)
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
        self.obj.set_pose(Pose(xyz, q))

    def _initialize_task(self, max_trials=100, verbose=False):
        obj_pos = self.obj.pose.p

        # Sample a goal position far enough from the object
        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            goal_z = self._episode_rng.uniform(0, 0.5) + obj_pos[2]
            goal_pos = np.hstack([goal_xy, goal_z])
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

@register_env("PickCube-v1", max_episode_steps=200)
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
                qvel = self.agent.robot.get_qvel()[:-2]
                static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
                reward += static_reward

        return reward

@register_env("PickCube-v2", max_episode_steps=100)
class PickCubeEnv_v2(PickCubeEnv_v1):
    # obs with is_grasp
    def _get_obs_extra(self) -> OrderedDict:
        ret = super()._get_obs_extra()
        ret['is_grasped'] = float(self.agent.check_grasp(self.obj))
        return ret
    
@register_env("PickCube-v3", max_episode_steps=100)
class PickCubeEnv_v3(PickCubeEnv_v2):
    # decrease sampling fields for both actors and task's goal
    def _initialize_actors(self):
        xy = self._episode_rng.uniform(-0.05, 0.05, [2])
        xyz = np.hstack([xy, self.cube_half_size[2]])
        q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)
        self.obj.set_pose(Pose(xyz, q))

    def _initialize_task(self, max_trials=100, verbose=False):
        obj_pos = self.obj.pose.p

        # Sample a goal position far enough from the object
        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.05, 0.05, [2])
            goal_z = self._episode_rng.uniform(0, 0.25) + obj_pos[2]
            goal_pos = np.hstack([goal_xy, goal_z])
            if np.linalg.norm(goal_pos - obj_pos) > self.min_goal_dist:
                if verbose:
                    print(f"Found a valid goal at {i}-th trial")
                break

        self.goal_pos = goal_pos
        self.goal_site.set_pose(Pose(self.goal_pos))

@register_env("PickCube-v4", max_episode_steps=100)
class PickCubeEnv_v4(PickCubeEnv_v3):
    def _get_obs_agent(self):
        """Remove gripper's vel and base pose."""
        proprioception = self.agent.get_proprioception()
        proprioception['qvel'] = proprioception['qvel'][:-2]
        return proprioception

@register_env("GraspCube-v0", max_episode_steps=100)
class GraspCubeEnv_v0(PickCubeEnv_v4):
    def _get_obs_extra(self) -> OrderedDict:
        """
            Delete unneeded obs for grasp a cube
        """
        ret = super()._get_obs_extra()
        del ret['goal_pos']
        del ret['tcp_to_goal_pos']
        del ret['obj_to_goal_pos']
        del ret['is_grasped']

        return ret

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 2
            return reward

        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward

        return reward
    
    def evaluate(self, **kwargs):
        is_grasped = self.agent.check_grasp(self.obj)
        # is_robot_static = self.check_robot_static()
        return dict(
            is_grasped=is_grasped,
            # is_robot_static=is_robot_static,
            success=is_grasped,
        )
    
@register_env("GraspCube-v1", max_episode_steps=100)
class GraspCubeEnv_v1(GraspCubeEnv_v0):
    def _get_obs_extra(self) -> OrderedDict:
        ret = super()._get_obs_extra()
        obs_noise = self._episode_rng.uniform(-0.01, 0.01, size=2)
        ret['obj_pose'][0:2] += obs_noise
        ret['tcp_to_obj_pos'][0:2] += obs_noise

        return ret
    
@register_env("GraspCubeY-v0", max_episode_steps=100)
class GraspCubeYEnv_v0(GraspCubeEnv_v1):
    def _initialize_actors(self):
        y = self._episode_rng.uniform(-0.05, 0.05, [2])
        xyz = np.hstack([np.zeros([0]), y, self.cube_half_size[2]])
        q = [1, 0, 0, 0]
        self.obj.set_pose(Pose(xyz, q))

    def _build_cube(self, half_size):
        builder: sapien.ArticulationBuilder = self._scene.create_articulation_builder()
        box: sapien.LinkBuilder = builder.create_link_builder()
        box.set_name('box')
        color_box=(1, 0, 0)
        render_material_box = self._renderer.create_material()
        render_material_box.set_base_color(np.hstack([color_box, 1.0]))
        box.add_box_collision(half_size=half_size)
        box.add_box_visual(half_size=half_size, material=render_material_box)

        front_slice = builder.create_link_builder(box)
        front_slice.set_name('front_slice')
        color_box=(0, 0, 1)
        render_material_box = self._renderer.create_material()
        render_material_box.set_base_color(np.hstack([color_box, 1.0]))
        front_slice.add_box_collision(pose=Pose(p=np.array([half_size[0], 0, -0.005])), half_size=np.array([0.001, 0.005, 0.005]))
        front_slice.add_box_visual(pose=Pose(p=np.array([half_size[0], 0, -0.005])), half_size=np.array([0.001, 0.005, 0.005]), material=render_material_box)

        back_slice = builder.create_link_builder(box)
        back_slice.set_name('back_slice')
        color_box=(0, 0, 1)
        render_material_box = self._renderer.create_material()
        render_material_box.set_base_color(np.hstack([color_box, 1.0]))
        back_slice.add_box_collision(pose=Pose(p=np.array([-half_size[0], 0, -0.005])), half_size=np.array([0.001, 0.005, 0.005]))
        back_slice.add_box_visual(pose=Pose(p=np.array([-half_size[0], 0, -0.005])), half_size=np.array([0.001, 0.005, 0.005]), material=render_material_box)

        cube = builder.build()
        cube.set_name('cube')
        return cube
    
    def check_articulation_grasp(self, articulation: sapien.ArticulationBase, min_impulse=1e-6, max_angle=85):
        contacts = self._scene.get_contacts()

        front_slice: sapien.LinkBase = get_entity_by_name(articulation.get_links(), 'front_slice')
        back_slice: sapien.LinkBase = get_entity_by_name(articulation.get_links(), 'back_slice')

        limpulse = get_pairwise_contact_impulse(contacts, self.agent.finger1_link, front_slice)
        rimpulse = get_pairwise_contact_impulse(contacts, self.agent.finger2_link, back_slice)
    
        ldirection = self.agent.finger1_link.pose.to_transformation_matrix()[:3, 1]
        rdirection = -self.agent.finger2_link.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection, limpulse)
        rangle = compute_angle_between(rdirection, rimpulse)

        lflag = (
            np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
        )
        rflag = (
            np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
        )

        return all([lflag, rflag])

    def evaluate(self, **kwargs):
        is_grasp = self.check_articulation_grasp(self.obj)

        return dict(is_grasp=float(is_grasp), success=float(is_grasp))


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

@register_env("LiftCube-v1", max_episode_steps=200)
class LiftCubeEnv_v1(LiftCubeEnv):
    # better reward
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5 # this should be larger than 2.25, otherwise robot will not learn to be static
            return reward

        # reaching reward
        gripper_pos = self.tcp.get_pose().p
        obj_pos = self.obj.get_pose().p
        dist = np.linalg.norm(gripper_pos - obj_pos)
        reaching_reward = 1 - np.tanh(5 * dist)
        reward += reaching_reward

        is_grasped = self.agent.check_grasp(self.obj)

        # grasp reward
        if is_grasped:
            reward += 1

        # lifting reward
        if is_grasped:
            lifting_reward = self.obj.pose.p[2] - self.cube_half_size[2]
            lifting_reward = min(lifting_reward / self.goal_height, 1.0)
            reward += lifting_reward

            # static reward
            if self.check_obj_placed():
                qvel = self.agent.robot.get_qvel()[:-2]
                static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))
                reward += static_reward

        return reward