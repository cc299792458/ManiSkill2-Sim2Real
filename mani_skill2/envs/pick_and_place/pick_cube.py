from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose

from .base_env import StationaryManipulationEnv


@register_env("PickCube-v0", max_episode_steps=200)
class PickCubeEnv(StationaryManipulationEnv):
    goal_thresh = 0.025
    min_goal_dist = 0.05

    def __init__(self, *args, obj_init_rot_z=True, size_range=0.0, **kwargs):
        self.obj_init_rot_z = obj_init_rot_z
        self.size_range = size_range
        self.org_half_cube_size = 0.02
        half_cube_size = self.org_half_cube_size
        self.cube_half_size = np.array([half_cube_size] * 3, np.float32)  # (chichu) change the half size of cube from 0.02 to 0.049/2 to align the real cube.
        self.last_obj_to_goal_dist = 0
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.cube_half_size[2] = self.org_half_cube_size
        self.obj = self._build_cube(self.cube_half_size)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_actors(self):
        if self.size_range != 0.0:
            self._actors.remove(self.obj)
            self._scene.remove_actor(self.obj)
            random_size = self._episode_rng.uniform(0, self.size_range)
            half_cube_size = self.org_half_cube_size + random_size
            self.cube_half_size = np.array([half_cube_size] * 3, np.float32)
            self.cube_half_size[2] = self.org_half_cube_size
            self.obj = self._build_cube(self.cube_half_size)
            self._actors.append(self.obj)

        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        xyz = np.hstack([xy, self.cube_half_size[2]])
        q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)
        # NOTE(chichu): fixed to a certain pose when evaluate on real robot with simulation.
        if self.fix_task_configuration:
            xyz = np.array([0.0, 0.0, self.cube_half_size[2]])
            ori = 0
            q = euler2quat(0, 0, ori)
        self.obj.set_pose(Pose(xyz, q))

    def _initialize_task(self, max_trials=100, verbose=False):
        obj_pos = self.obj.pose.p
        # Sample a goal position far enough from the object
        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.05, 0.05, [2])
            goal_z = self._episode_rng.uniform(0, 0.2) + obj_pos[2]
            goal_pos = np.hstack([goal_xy, goal_z])
            # NOTE(chichu): set to a fixed point when evaluate real robot with simulation
            if self.fix_task_configuration:
                goal_pos = np.array([0.0, 0.0, 0.1]) + obj_pos[2]
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
            obj_pose = vectorize_pose(self.obj.pose)
            tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
            obj_to_goal_pos = self.goal_pos - self.obj.pose.p
            if self.obs_noise != 0.0:
                xy_noise = self.generate_noise_for_pos(size=2)
                obj_pose[0:2] += xy_noise
                tcp_to_obj_pos[0:2] += xy_noise
                obj_to_goal_pos[0:2] -= xy_noise
            obs.update(
                tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                obj_pose=obj_pose,
                tcp_to_obj_pos=tcp_to_obj_pos,
                obj_to_goal_pos=obj_to_goal_pos,
                # Add if the cube is grasped
                obj_grasped=float(self.agent.check_grasp(self.obj)),
            )
            if self.ee_move_independently:
                obs.update(reached=float(self.check_reached()),)
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
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0
        if info["success"]:
            reward += 6
            return reward
        if info["time_out"]:
            reward -= 3
        # if info["ee_constraint_break"]:
        #     reward -= 8
        # #####----- Angular velocity penalty -----#####
        # obj_angvel = self.obj.angular_velocity
        # obj_angvel_norm = np.linalg.norm(obj_angvel)
        # if obj_angvel_norm > 0.5:
        #     reward -= 0.5 
        #####----- Reach reward -----#####
        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward
        #####----- Grasp rotate reward -----#####
        grasp_rot_loss_fxn = lambda A: np.tanh(np.trace(A.T @ A))  # trace(A.T @ A) has range [0,8] for A being difference of rotation matrices
        tcp_pose_wrt_obj = self.obj.pose.inv() * self.tcp.pose
        tcp_rot_wrt_obj = tcp_pose_wrt_obj.to_transformation_matrix()[:3, :3]
        gt_rots = [
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
            np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        ]
        grasp_rot_loss = min([grasp_rot_loss_fxn(x - tcp_rot_wrt_obj) for x in gt_rots])
        reward += 1 - grasp_rot_loss
        #####----- Grasped reward -----#####
        is_grasped = self.agent.check_grasp(self.obj) # remove max_angle=30 yeilds much better performance
        if is_grasped:
            reward += 1
            #####----- Rotate reward -----#####
            obj_quat = self.obj.pose.q
            obj_euler = np.abs(quat2euler(obj_quat))
            obj_euler_xy = (obj_euler[0]+obj_euler[1])
            reward += (1 - np.tanh(obj_euler_xy)) / 2
            #####----- Reach reward 2 -----#####
            obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
            if obj_to_goal_dist < self.last_obj_to_goal_dist:
                place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
                reward += place_reward
            self.last_obj_to_goal_dist = obj_to_goal_dist
            #####----- Static reward -----#####
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
