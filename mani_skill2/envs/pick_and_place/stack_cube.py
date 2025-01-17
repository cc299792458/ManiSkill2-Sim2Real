from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat, quat2euler

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import check_actor_static, vectorize_pose

from .base_env import StationaryManipulationEnv


class UniformSampler:
    """Uniform placement sampler.

    Args:
        ranges: ((low1, low2, ...), (high1, high2, ...))
        rng (np.random.RandomState): random generator
    """

    def __init__(
        self, ranges: Tuple[List[float], List[float]], rng: np.random.RandomState
    ) -> None:
        assert len(ranges) == 2 and len(ranges[0]) == len(ranges[1])
        self._ranges = ranges
        self._rng = rng
        self.fixtures = []

    def sample(self, radius, max_trials, append=True, verbose=False):
        """Sample a position.

        Args:
            radius (float): collision radius.
            max_trials (int): maximal trials to sample.
            append (bool, optional): whether to append the new sample to fixtures. Defaults to True.
            verbose (bool, optional): whether to print verbosely. Defaults to False.

        Returns:
            np.ndarray: a sampled position.
        """
        if len(self.fixtures) == 0:
            pos = self._rng.uniform(*self._ranges)
        else:
            fixture_pos = np.array([x[0] for x in self.fixtures])
            fixture_radius = np.array([x[1] for x in self.fixtures])
            for i in range(max_trials):
                pos = self._rng.uniform(*self._ranges)
                dist = np.linalg.norm(pos - fixture_pos, axis=1)
                if np.all(dist > fixture_radius + radius):
                    if verbose:
                        print(f"Found a valid sample at {i}-th trial")
                    break
            else:
                if verbose:
                    print("Fail to find a valid sample!")
        if append:
            self.fixtures.append((pos, radius))
        return pos


@register_env("StackCube-v0", max_episode_steps=200)
class StackCubeEnv(StationaryManipulationEnv):
    def __init__(self, *args, robot="xarm7_d435", robot_init_qpos_noise=0.02, 
                domain_rand_params = dict(size_range=0.005, fric_range=[0.5, 1.5], obs_noise=0.0025), **kwargs):
        if domain_rand_params is not None:
            self.domain_rand = True
            self.size_range = domain_rand_params['size_range']
            self.obs_noise = domain_rand_params['obs_noise']
            self.fric_range = domain_rand_params['fric_range']
        else:
            self.domain_rand = False
            self.size_range = 0.0
            self.obs_noise = 0.0
        self.org_half_cube_size = 0.02
        super().__init__(*args, robot=robot, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    def _get_default_scene_config(self, sim_params, enable_tgs):
        scene_config = super()._get_default_scene_config(sim_params, enable_tgs)
        scene_config.enable_pcm = True
        return scene_config

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        self.box_half_size = np.float32([0.02] * 3)
        self.cubeA = self._build_cube(self.box_half_size, color=(1, 0, 0), name="cubeA")
        self.cubeB = self._build_cube(
            self.box_half_size, color=(0, 1, 0), name="cubeB", static=False
        )

    def _initialize_actors(self):
        if self.domain_rand:
            # remove original object
            self._actors.remove(self.cubeA)
            self._scene.remove_actor(self.cubeA)
            # sample properties of new object
            random_size = self._episode_rng.uniform(-self.size_range, self.size_range)
            half_cube_size = self.org_half_cube_size + random_size
            self.cube_half_size = np.array([half_cube_size] * 3, np.float32)
            self.cube_half_size[2] = self.org_half_cube_size
            self.friction = self._episode_rng.uniform(self.fric_range[0], self.fric_range[1])
            physcial_mat = self._scene.create_physical_material(static_friction=self.friction, dynamic_friction=self.friction, restitution=0.0)
            # create new object
            self.cubeA = self._build_cube(self.cube_half_size, physical_material=physcial_mat)
            self._actors.append(self.cubeA)
        
        # decrease region size.
        xy = self._episode_rng.uniform(-0.05, 0.05, [2])
        region = [[-0.05, -0.1], [0.05, 0.1]]
        sampler = UniformSampler(region, self._episode_rng)
        radius = np.linalg.norm(self.box_half_size[:2]) + 0.001
        cubeA_xy = xy + sampler.sample(radius, 100)
        cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

        cubeA_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cubeB_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        z = self.box_half_size[2]
        if self.fix_task_configuration:
            cubeA_xy = np.array([0.0, 0.0])
            cubeB_xy = np.array([0.05, 0.05])
            cubeA_quat = cubeB_quat = np.array([1.0, 0.0, 0.0, 0.0])
        cubeA_pose = sapien.Pose([cubeA_xy[0], cubeA_xy[1], z], cubeA_quat)
        cubeB_pose = sapien.Pose([cubeB_xy[0], cubeB_xy[1], z], cubeB_quat)

        self.cubeA.set_pose(cubeA_pose)
        self.cubeB.set_pose(cubeB_pose)

    def _get_obs_extra(self):
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                cubeA_pose=vectorize_pose(self.cubeA.pose),
                cubeB_pose=vectorize_pose(self.cubeB.pose),
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                # Add some binary information to facilitate training.
                is_cubeA_grasped=float(self.agent.check_grasp(self.cubeA)),
                is_cubeA_on_cubeB=float(self._check_cubeA_on_cubeB()),
            )
        return obs

    def _check_cubeA_on_cubeB(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (np.linalg.norm(offset[:2]) <= np.linalg.norm(self.box_half_size[:2]) + 0.005)
        z_flag = np.abs(offset[2] - self.box_half_size[2] * 2) <= 0.005
        return bool(xy_flag and z_flag)

    def evaluate(self, **kwargs):
        is_cubeA_on_cubeB = self._check_cubeA_on_cubeB()
        is_cubeA_static = check_actor_static(self.cubeA)
        is_cubaA_grasped = self.agent.check_grasp(self.cubeA)
        success = is_cubeA_on_cubeB and is_cubeA_static and (not is_cubaA_grasped)

        return {
            "is_cubaA_grasped": is_cubaA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "cubeA_vel": np.linalg.norm(self.cubeA.velocity),
            "cubeA_ang_vel": np.linalg.norm(self.cubeA.angular_velocity),
            "success": success,
        }

    def compute_dense_reward(self, info, **kwargs):
        gripper_width = (
            self.agent.robot.get_qlimits()[-1, 1] * 2
        )  # NOTE: hard-coded with panda
        reward = 0.0

        if info["success"]:
            reward = 15.0
        else:
            # grasp pose rotation reward
            grasp_rot_loss_fxn = lambda A: np.tanh(
                1 / 8 * np.trace(A.T @ A)
            )  # trace(A.T @ A) has range [0,8] for A being difference of rotation matrices
            tcp_pose_wrt_cubeA = self.cubeA.pose.inv() * self.tcp.pose
            tcp_rot_wrt_cubeA = tcp_pose_wrt_cubeA.to_transformation_matrix()[:3, :3]
            gt_rots = [
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            ]
            grasp_rot_loss = min(
                [grasp_rot_loss_fxn(x - tcp_rot_wrt_cubeA) for x in gt_rots]
            )
            reward += 1 - grasp_rot_loss

            cubeB_vel_penalty = np.linalg.norm(self.cubeB.velocity) + np.linalg.norm(
                self.cubeB.angular_velocity
            )
            reward -= cubeB_vel_penalty

            # reaching object reward
            tcp_pose = self.tcp.pose.p
            cubeA_pos = self.cubeA.pose.p
            cubeA_to_tcp_dist = np.linalg.norm(tcp_pose - cubeA_pos)
            reaching_reward = 1 - np.tanh(3.0 * cubeA_to_tcp_dist)
            reward += reaching_reward

            # check if cubeA is on cubeB
            cubeA_pos = self.cubeA.pose.p
            cubeB_pos = self.cubeB.pose.p
            goal_xyz = np.hstack(
                [cubeB_pos[0:2], cubeB_pos[2] + self.box_half_size[2] * 2]
            )
            cubeA_on_cubeB = (
                np.linalg.norm(goal_xyz[:2] - cubeA_pos[:2])
                < self.box_half_size[0] * 0.8
            )
            cubeA_on_cubeB = cubeA_on_cubeB and (
                np.abs(goal_xyz[2] - cubeA_pos[2]) <= 0.005
            )
            if cubeA_on_cubeB:
                reward = 10.0
                # ungrasp reward
                is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
                if not is_cubeA_grasped:
                    reward += 2.0
                else:
                    reward = (
                        reward
                        + 2.0 * np.sum(self.agent.robot.get_qpos()[-2:]) / gripper_width
                    )
            else:
                # grasping reward
                is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
                if is_cubeA_grasped:
                    reward += 1.0

                # reaching goal reward, ensuring that cubeA has appropriate height during this process
                if is_cubeA_grasped:
                    cubeA_to_goal = goal_xyz - cubeA_pos
                    # cubeA_to_goal_xy_dist = np.linalg.norm(cubeA_to_goal[:2])
                    cubeA_to_goal_dist = np.linalg.norm(cubeA_to_goal)
                    appropriate_height_penalty = np.maximum(
                        np.maximum(2 * cubeA_to_goal[2], 0.0),
                        np.maximum(2 * (-0.02 - cubeA_to_goal[2]), 0.0),
                    )
                    reaching_reward2 = 2 * (
                        1 - np.tanh(5.0 * appropriate_height_penalty)
                    )
                    # qvel_penalty = np.sum(np.abs(self.agent.robot.get_qvel())) # prevent the robot arm from moving too fast
                    # reaching_reward2 -= 0.0003 * qvel_penalty
                    # if appropriate_height_penalty < 0.01:
                    reaching_reward2 += 4 * (1 - np.tanh(5.0 * cubeA_to_goal_dist))
                    reward += np.maximum(reaching_reward2, 0.0)

        return reward
        
@register_env("StackCube-v1", max_episode_steps=100)
class StackCubeEnv_v1(StackCubeEnv):
    def reaching_reward(self):
        # reaching object reward
        tcp_pose = self.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = np.linalg.norm(tcp_pose - cubeA_pos)
        return 1 - np.tanh(5 * cubeA_to_tcp_dist)

    def place_reward(self):
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = np.hstack([cubeB_pos[0:2], cubeB_pos[2] + self.box_half_size[2] * 2])
        cubeA_to_goal_dist = np.linalg.norm(goal_xyz - cubeA_pos)
        reaching_reward2 = 1 - np.tanh(5.0 * cubeA_to_goal_dist)
        return reaching_reward2

    def ungrasp_reward(self):
        gripper_width = (
            self.agent.robot.get_qlimits()[-1, 1] * 2
        )  # NOTE: hard-coded with panda
        # ungrasp reward
        is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
        if not is_cubeA_grasped:
            reward = 1.0
        else:
            reward = np.sum(self.agent.robot.get_qpos()[-2:]) / gripper_width

        v = np.linalg.norm(self.cubeA.velocity)
        av = np.linalg.norm(self.cubeA.angular_velocity)
        static_reward = 1 - np.tanh(v*10 + av)
        
        return (reward + static_reward) / 2.0
        
    def compute_dense_reward(self, info, **kwargs):

        if info["success"]:
            reward = 8
        elif self._check_cubeA_on_cubeB():
            reward = 6 + self.ungrasp_reward()
        elif self.agent.check_grasp(self.cubeA):
            reward = 4 + self.place_reward()
        else:
            reward = 2 + self.reaching_reward()

        # reward = reward - 9.0

        return reward

# @register_env("StackCube-v2", max_episode_steps=100)
# class StackCubeEnv_v2(StackCubeEnv):
#     def reaching_reward(self):
#         # reaching object reward
#         tcp_pose = self.tcp.pose.p
#         cubeA_pos = self.cubeA.pose.p
#         cubeA_to_tcp_dist = np.linalg.norm(tcp_pose - cubeA_pos)
#         return 1 - np.tanh(5 * cubeA_to_tcp_dist)

#     def place_reward(self):
#         cubeA_pos = self.cubeA.pose.p
#         cubeB_pos = self.cubeB.pose.p
#         goal_xyz = np.hstack([cubeB_pos[0:2], cubeB_pos[2] + self.box_half_size[2] * 2])
#         cubeA_to_goal_dist = np.linalg.norm(goal_xyz - cubeA_pos)
#         reaching_reward2 = 1 - np.tanh(5.0 * cubeA_to_goal_dist)
#         return reaching_reward2

#     def ungrasp_reward(self):
#         gripper_width = (
#             self.agent.robot.get_qlimits()[-1, 1]
#         )  # NOTE: hard-coded with xarm, reduced-gripper
#         # ungrasp reward
#         is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
#         if not is_cubeA_grasped:
#             reward = 1.0
#         else:
#             reward = self.agent.robot.get_qpos()[-1] / gripper_width

#         v = np.linalg.norm(self.cubeA.velocity)
#         av = np.linalg.norm(self.cubeA.angular_velocity)
#         static_reward = 1 - np.tanh(v*10 + av)
        
#         return (reward + static_reward) / 2.0
    
#     # def ungrasp_reward(self):
#     #     gripper_width = (
#     #         self.agent.robot.get_qlimits()[-1, 1] * 2
#     #     )  # NOTE: hard-coded with panda
#     #     # ungrasp reward
#     #     is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
#     #     if not is_cubeA_grasped:
#     #         reward = 1.0
#     #     else:
#     #         reward = np.sum(self.agent.robot.get_qpos()[-2:]) / gripper_width
    
#     #     v = np.linalg.norm(self.cubeA.velocity)
#     #     av = np.linalg.norm(self.cubeA.angular_velocity)
#     #     static_reward = 1 - np.tanh(v*10 + av)

#     #     return (reward + static_reward) / 2.0
        
#     def compute_dense_reward(self, info, **kwargs):

#         if info["success"]:
#             reward = 8
#         elif self._check_cubeA_on_cubeB():
#             reward = 6 + self.ungrasp_reward()
#         elif self.agent.check_grasp(self.cubeA):
#             reward = 4 + self.place_reward()
#         else:
#             reward = 2 + self.reaching_reward()

#         # reward = reward - 9.0
#         if info["time_out"]:
#             reward -= 3

#         return reward

@register_env("StackCube-v2", max_episode_steps=100)
class StackCubeEnv_v2(StackCubeEnv_v1):
    def ungrasp_reward(self):
        gripper_width = (
            self.agent.robot.get_qlimits()[-1, 1]
        )  # NOTE: hard-coded with xarm, reduced-gripper
        # ungrasp reward
        is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
        if not is_cubeA_grasped:
            reward = 1.0
        else:
            reward = self.agent.robot.get_qpos()[-1] / gripper_width

        v = np.linalg.norm(self.cubeA.velocity)
        av = np.linalg.norm(self.cubeA.angular_velocity)
        static_reward = 1 - np.tanh(v*10 + av)
        
        return (reward + static_reward) / 2.0
    
    def compute_dense_reward(self, info, **kwargs):

        reward = super().compute_dense_reward(info, **kwargs)
        # reward = reward - 9.0
        if info["time_out"]:
            reward -= 2

        return reward

@register_env("StackCube-v3", max_episode_steps=100)
class StackCubeEnv_v3(StackCubeEnv_v2):
    def _get_obs_agent(self):
        """Remove gripper's vel and base pose."""
        proprioception = self.agent.get_proprioception()
        proprioception['qvel'] = proprioception['qvel'][:-2]
        return proprioception
    
# @register_env("StackCube-v3", max_episode_steps=50)
# class StackCubeEnv_v3(StackCubeEnv):
#     def generate_noise_for_pos(self, size):
#         noise = np.random.uniform(-self.obs_noise, self.obs_noise, size=size)
#         return noise

#     def _get_obs_extra(self):
#         obs = OrderedDict(
#             tcp_pose=vectorize_pose(self.tcp.pose),
#         )
#         cubeA_pose = vectorize_pose(self.cubeA.pose)
#         cubeB_pose = vectorize_pose(self.cubeB.pose)
#         tcp_to_cubeA_pos = self.cubeA.pose.p - self.tcp.pose.p
#         tcp_to_cubeB_pos = self.cubeB.pose.p - self.tcp.pose.p
#         cubeA_to_cubeB_pos = self.cubeB.pose.p - self.cubeA.pose.p
#         if self.domain_rand:
#             cubeA_xy_noise = self.generate_noise_for_pos(size=2)
#             cubeB_xy_noise = self.generate_noise_for_pos(size=2)
#             cubeA_pose[0:2] += cubeA_xy_noise
#             cubeB_pose[0:2] += cubeB_xy_noise
#             tcp_to_cubeA_pos[0:2] += cubeA_xy_noise
#             tcp_to_cubeB_pos[0:2] += cubeB_xy_noise
#             cubeA_to_cubeB_pos[0:2] += (cubeB_xy_noise - cubeA_xy_noise)
#         if self._obs_mode in ["state", "state_dict"]:
#             obs.update(
#                 cubeA_pose=cubeA_pose,
#                 cubeB_pose=cubeB_pose,
#                 tcp_to_cubeA_pos=tcp_to_cubeA_pos,
#                 tcp_to_cubeB_pos=tcp_to_cubeB_pos,
#                 cubeA_to_cubeB_pos=cubeA_to_cubeB_pos,
#                 # Add some binary information to facilitate training.
#                 cubeA_vel=np.linalg.norm(self.cubeA.velocity),
#                 cubeA_ang_vel=np.linalg.norm(self.cubeA.angular_velocity),
#                 is_cubeA_grasped=float(self.agent.check_grasp(self.cubeA)),
#                 is_cubeA_on_cubeB=float(self._check_cubeA_on_cubeB()),
#                 is_cubeA_lift=float(self._check_cubeA_lift()),
#                 is_cubeA_above_cubeB=float(self._check_cubeA_above_cubeB())
#             )
#         return obs
    
#     def evaluate(self, **kwargs):
#         is_cubeA_on_cubeB = self._check_cubeA_on_cubeB()
#         is_cubeA_static = check_actor_static(self.cubeA)
#         is_cubaA_grasped = self.agent.check_grasp(self.cubeA)
#         is_cubeA_lift = self._check_cubeA_lift()
#         is_cubeA_above_cubeB = self._check_cubeA_above_cubeB()
#         success = is_cubeA_on_cubeB and is_cubeA_static and (not is_cubaA_grasped)
        
#         return {
#             "is_cubaA_grasped": is_cubaA_grasped,
#             "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
#             "is_cubeA_static": is_cubeA_static,
#             "is_cubeA_lift": is_cubeA_lift,
#             "is_cubeA_above_cubeB": is_cubeA_above_cubeB,
#             "cubeA_vel": np.linalg.norm(self.cubeA.velocity),
#             "cubeA_ang_vel": np.linalg.norm(self.cubeA.angular_velocity),
#             "success": success,
#         }
    
#     def _check_cubeA_lift(self, height=0.08):
#         cubeA_pos = self.cubeA.pose.p
#         cubeA_to_height_dist = np.abs(height - cubeA_pos[2])
#         return bool(cubeA_to_height_dist < 0.01)
    
#     def _check_cubeA_above_cubeB(self):
#         cubeA_pos = self.cubeA.pose.p
#         cubeB_pos = self.cubeB.pose.p
#         cubeA_to_cubeB_dist = np.linalg.norm(cubeB_pos[:2] - cubeA_pos[:2])
#         return bool(cubeA_to_cubeB_dist < 0.01)
    
#     def reaching_reward(self):
#         # reaching object reward
#         tcp_pose = self.tcp.pose.p
#         cubeA_pos = self.cubeA.pose.p
#         cubeA_to_tcp_dist = np.linalg.norm(tcp_pose - cubeA_pos)
#         reaching_reward = 1 - np.tanh(5 * cubeA_to_tcp_dist)

#         return reaching_reward
    
#     def grasp_rotate_reward(self):
#         grasp_rot_loss_fxn = lambda A: np.tanh(np.trace(A.T @ A))  # trace(A.T @ A) has range [0,8] for A being difference of rotation matrices
#         tcp_pose_wrt_cubeA = self.cubeA.pose.inv() * self.tcp.pose
#         tcp_rot_wrt_cubeA = tcp_pose_wrt_cubeA.to_transformation_matrix()[:3, :3]
#         gt_rots = [
#             np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
#             np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
#             np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
#             np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
#         ]
#         grasp_rot_loss = min([grasp_rot_loss_fxn(x - tcp_rot_wrt_cubeA) for x in gt_rots])
#         reward = 1 - grasp_rot_loss

#         return reward

#     def lift_reward(self, height=0.08):
#         cubeA_pos = self.cubeA.pose.p
#         cubeB_pos = self.cubeB.pose.p
#         goal_pos = np.hstack([cubeB_pos[0:2], np.array([height])])
#         cubeA_to_subgoal_dist = np.linalg.norm(goal_pos - cubeA_pos)
#         return 1 - np.tanh(5 * cubeA_to_subgoal_dist)
    
#     def move_reward(self):
#         cubeA_pos = self.cubeA.pose.p
#         cubeB_pos = self.cubeB.pose.p
#         cubeA_to_cubeB_dist = np.linalg.norm(cubeB_pos[:2] - cubeA_pos[:2])
#         move_reward = 1 - np.tanh(5.0 * cubeA_to_cubeB_dist)
#         return move_reward

#     def place_reward(self):
#         cubeA_pos = self.cubeA.pose.p
#         cubeB_pos = self.cubeB.pose.p
#         goal_xyz = np.hstack([cubeB_pos[0:2], cubeB_pos[2] + self.box_half_size[2] * 2])
#         cubeA_to_goal_dist = np.linalg.norm(goal_xyz - cubeA_pos)
#         reaching_reward2 = 1 - np.tanh(5.0 * cubeA_to_goal_dist)
#         return reaching_reward2

#     def ungrasp_reward(self):
#         gripper_width = (
#             self.agent.robot.get_qlimits()[-1, 1]
#         )  # NOTE: hard-coded with xarm, reduced-gripper
#         # ungrasp reward
#         is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
#         if not is_cubeA_grasped:
#             reward = 1.0
#         else:
#             reward = self.agent.robot.get_qpos()[-1] / gripper_width

#         v = np.linalg.norm(self.cubeA.velocity)
#         av = np.linalg.norm(self.cubeA.angular_velocity)
#         static_reward = 1 - np.tanh(v*10 + av)
        
#         return (reward + static_reward) / 2.0
    
#     # def unrotate_reward(self):
#     #     cubeA_quat = self.cubeA.pose.q
#     #     cubeA_euler = np.abs(quat2euler(cubeA_quat))

#     #     roll, pitch = cubeA_euler[0], cubeA_euler[1]
#     #     unrotate_reward = 1 - np.clip((roll+pitch)/(np.pi/4), a_min=0, a_max=1)

#     #     return unrotate_reward

#     def compute_dense_reward(self, info, **kwargs):

#         if info["success"]:
#             reward = 6
#         elif self._check_cubeA_on_cubeB():
#             reward = 5 + self.ungrasp_reward()
#         elif self._check_cubeA_above_cubeB():
#             reward = 4 + self.place_reward()
#         elif self._check_cubeA_lift():
#             reward = 3 + self.move_reward()
#         elif self.agent.check_grasp(self.cubeA):
#             reward = 2 + self.lift_reward()
#         else:
#             reward = self.reaching_reward() + self.grasp_rotate_reward()

#         # if self.agent.check_grasp(self.cubeA):
#         #     reward += self.unrotate_reward()

#         cubeB_vel_penalty = 10 * np.linalg.norm(self.cubeB.velocity) + \
#                         np.linalg.norm(self.cubeB.angular_velocity)
#         reward += 1 - np.tanh(cubeB_vel_penalty)

#         # reward = reward - 9.0
#         if info["time_out"]:
#             reward = -3

#         return np.clip(reward / 7, a_min=-1, a_max=1)

