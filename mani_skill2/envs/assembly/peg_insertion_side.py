from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import hex2rgba, look_at, vectorize_pose

from .base_env import StationaryManipulationEnv


@register_env("PegInsertionSide-v0", max_episode_steps=200)
class PegInsertionSideEnv(StationaryManipulationEnv):
    _clearance = 0.003

    def reset(self, reconfigure=True, **kwargs):
        return super().reset(reconfigure=reconfigure, **kwargs)

    def _build_box_with_hole(
        self, inner_radius, outer_radius, depth, center=(0, 0), name="box_with_hole"
    ):
        builder = self._scene.create_actor_builder()
        thickness = (outer_radius - inner_radius) * 0.5
        # x-axis is hole direction
        half_center = [x * 0.5 for x in center]
        half_sizes = [
            [depth, thickness - half_center[0], outer_radius],
            [depth, thickness + half_center[0], outer_radius],
            [depth, outer_radius, thickness - half_center[1]],
            [depth, outer_radius, thickness + half_center[1]],
        ]
        offset = thickness + inner_radius
        poses = [
            Pose([0, offset + half_center[0], 0]),
            Pose([0, -offset + half_center[0], 0]),
            Pose([0, 0, offset + half_center[1]]),
            Pose([0, 0, -offset + half_center[1]]),
        ]

        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#FFD289"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5

        for (half_size, pose) in zip(half_sizes, poses):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)
        return builder.build_static(name)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        # peg
        # length, radius = 0.1, 0.02
        length = self._episode_rng.uniform(0.075, 0.125)
        radius = self._episode_rng.uniform(0.015, 0.025)
        self.peg_half_length = length
        self.peg_radius = radius
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[length, radius, radius])

        # peg head
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EC7357"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        # peg tail
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EDF6F9"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([-length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        self.peg = builder.build("peg")
        self.peg_head_offset = Pose([length, 0, 0])
        self.peg_half_size = np.float32([length, radius, radius])

        # box with hole
        center = 0.5 * (length - radius) * self._episode_rng.uniform(-1, 1, size=2)
        inner_radius, outer_radius, depth = radius + self._clearance, length, length
        self.box = self._build_box_with_hole(
            inner_radius, outer_radius, depth, center=center
        )
        self.box_hole_offset = Pose(np.hstack([0, center]))
        self.box_hole_radius = inner_radius

    def _initialize_actors(self):
        xy = self._episode_rng.uniform([-0.1, -0.3], [0.1, 0])
        pos = np.hstack([xy, self.peg_half_size[2]])
        ori = np.pi / 2 + self._episode_rng.uniform(-np.pi / 3, np.pi / 3)
        quat = euler2quat(0, 0, ori)
        self.peg.set_pose(Pose(pos, quat))

        xy = self._episode_rng.uniform([-0.05, 0.2], [0.05, 0.4])
        pos = np.hstack([xy, self.peg_half_size[0]])
        ori = np.pi / 2 + self._episode_rng.uniform(-np.pi / 8, np.pi / 8)
        quat = euler2quat(0, 0, ori)
        self.box.set_pose(Pose(pos, quat))

    def _initialize_agent(self):
        if self.robot_uid == "panda":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        elif self.robot_uid in ['xarm7', 'xarm7_d435']:
            if self.ee_type == 'reduced_gripper':
                qpos = np.array(
                        [0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2, 0.0446430, 0.0446430]
                    )
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.4639, 0.0, 0.0]))
        else:
            raise NotImplementedError(self.robot_uid)

    @property
    def peg_head_pos(self):
        return self.peg.pose.transform(self.peg_head_offset).p

    @property
    def peg_head_pose(self):
        return self.peg.pose.transform(self.peg_head_offset)

    @property
    def box_hole_pose(self):
        return self.box.pose.transform(self.box_hole_offset)

    def _initialize_task(self):
        self.goal_pos = self.box_hole_pose.p  # goal of peg head inside the hole
        # NOTE(jigu): The goal pose is computed based on specific geometries used in this task.
        # Only consider one side
        self.goal_pose = (
            self.box.pose * self.box_hole_offset * self.peg_head_offset.inv()
        )
        # self.peg.set_pose(self.goal_pose)

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                peg_pose=vectorize_pose(self.peg.pose),
                peg_half_size=self.peg_half_size,
                box_hole_pose=vectorize_pose(self.box_hole_pose),
                box_hole_radius=self.box_hole_radius,
            )
        return obs

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
        box_hole_pose = self.box_hole_pose
        peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[0]
        y_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[1] <= self.box_hole_radius
        )
        z_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[2] <= self.box_hole_radius
        )
        return (x_flag and y_flag and z_flag), peg_head_pos_at_hole

    def evaluate(self, **kwargs) -> dict:
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            return 25.0

        # grasp pose rotation reward
        tcp_pose_wrt_peg = self.peg.pose.inv() * self.tcp.pose
        tcp_rot_wrt_peg = tcp_pose_wrt_peg.to_transformation_matrix()[:3, :3]
        gt_rot_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        gt_rot_2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        grasp_rot_loss_fxn = lambda A: np.arcsin(
            np.clip(1 / (2 * np.sqrt(2)) * np.sqrt(np.trace(A.T @ A)), 0, 1)
        )
        grasp_rot_loss = np.minimum(
            grasp_rot_loss_fxn(gt_rot_1 - tcp_rot_wrt_peg),
            grasp_rot_loss_fxn(gt_rot_2 - tcp_rot_wrt_peg),
        ) / (np.pi / 2)
        rotated_properly = grasp_rot_loss < 0.2
        reward += 1 - grasp_rot_loss

        gripper_pos = self.tcp.pose.p
        tgt_gripper_pose = self.peg.pose
        offset = sapien.Pose(
            [-0.06, 0, 0]
        )  # account for panda gripper width with a bit more leeway
        tgt_gripper_pose = tgt_gripper_pose.transform(offset)
        if rotated_properly:
            # reaching reward
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - tgt_gripper_pose.p)
            reaching_reward = 1 - np.tanh(
                4.0 * np.maximum(gripper_to_peg_dist - 0.015, 0.0)
            )
            # reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(
                self.peg, max_angle=20
            )  # max_angle ensures that the gripper grasps the peg appropriately, not in a strange pose
            if is_grasped:
                reward += 2.0

            # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
            pre_inserted = False
            if is_grasped:
                peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
                peg_head_wrt_goal_yz_dist = np.linalg.norm(peg_head_wrt_goal.p[1:])
                peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
                peg_wrt_goal_yz_dist = np.linalg.norm(peg_wrt_goal.p[1:])
                if peg_head_wrt_goal_yz_dist < 0.01 and peg_wrt_goal_yz_dist < 0.01:
                    pre_inserted = True
                    reward += 3.0
                pre_insertion_reward = 3 * (
                    1
                    - np.tanh(
                        0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
                        + 4.5
                        * np.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
                    )
                )
                reward += pre_insertion_reward

            # insertion reward
            if is_grasped and pre_inserted:
                peg_head_wrt_goal_inside_hole = (
                    self.box_hole_pose.inv() * self.peg_head_pose
                )
                insertion_reward = 5 * (
                    1 - np.tanh(5.0 * np.linalg.norm(peg_head_wrt_goal_inside_hole.p))
                )
                reward += insertion_reward
        else:
            reward = reward - 10 * np.maximum(
                self.peg.pose.p[2] + self.peg_half_size[2] + 0.01 - self.tcp.pose.p[2],
                0.0,
            )
            reward = reward - 10 * np.linalg.norm(
                tgt_gripper_pose.p[:2] - self.tcp.pose.p[:2]
            )

        return reward

    def _register_cameras(self):
        cam_cfg = super()._register_cameras()
        cam_cfg.pose = look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return cam_cfg

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.pose = look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return cam_cfg

    def set_state(self, state):
        super().set_state(state)
        # NOTE(xuanlin): This way is specific to how we compute goals.
        # The general way is to handle variables explicitly
        self._initialize_task()

@register_env("PegInsertionSideFixed-v0", max_episode_steps=200)
class PegInsertionSideEnv_fixed(PegInsertionSideEnv):
    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        # peg
        length, radius = 0.1, 0.02
        self.peg_half_length = length
        self.peg_radius = radius
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[length, radius, radius])

        # peg head
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EC7357"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        # peg tail
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EDF6F9"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([-length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        self.peg = builder.build("peg")
        self.peg_head_offset = Pose([length, 0, 0])
        self.peg_half_size = np.float32([length, radius, radius])

        # box with hole
        center = np.zeros(2)
        inner_radius, outer_radius, depth = radius + self._clearance, length, length
        self.box = self._build_box_with_hole(
            inner_radius, outer_radius, depth, center=center
        )
        self.box_hole_offset = Pose(np.hstack([0, center]))
        self.box_hole_radius = inner_radius

    def _initialize_actors(self):
        xy = np.array([0, -0.15])
        pos = np.hstack([xy, self.peg_half_size[2]])
        ori = np.pi / 2
        quat = euler2quat(0, 0, ori)
        self.peg.set_pose(Pose(pos, quat))

        xy = np.array([0, 0.3])
        pos = np.hstack([xy, self.peg_half_size[0]])
        ori = np.pi / 2
        quat = euler2quat(0, 0, ori)
        self.box.set_pose(Pose(pos, quat))


@register_env("PegInsertionSideFixed_simple_rew-v0", max_episode_steps=200)
class PegInsertionSideEnv_fixed_simple_rew(PegInsertionSideEnv_fixed):
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 5.25 + 1
        else:
            # reaching reward
            gripper_pos = self.tcp.get_pose().p
            peg_pos = self.peg.get_pose().p
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - peg_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 + np.tanh(10.0 * (peg_head_pos_at_hole[0] + 0.015)) # (0, 2)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1)
                align_reward_z = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[2])) # (0, 1) 
                
                reward += insertion_reward + align_reward_y + align_reward_z

        return reward
    
@register_env("PegInsertionSideFixed_peg_ori-v0", max_episode_steps=200)
class PegInsertionSideEnv_fixed_peg_ori(PegInsertionSideEnv_fixed):
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 7.25 + 1
        else:
            # reaching reward
            gripper_pos = self.tcp.get_pose().p
            peg_pos = self.peg.get_pose().p
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - peg_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 + np.tanh(10.0 * (peg_head_pos_at_hole[0] + 0.015)) # (0, 2)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1)
                align_reward_z = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[2])) # (0, 1) 
                
                reward += insertion_reward + align_reward_y + align_reward_z

                peg_axis = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += cos * 2

        return reward
    
@register_env("PegInsertionSideFixed_grasp_offset-v0", max_episode_steps=200)
class PegInsertionSideEnv_fixed_grasp_offset(PegInsertionSideEnv_fixed):
    
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 7.25 + 1
        else:
            # reaching reward
            gripper_pos = self.tcp.get_pose().p
            peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
            head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
            grasp_pos = center_pos - (head_pos - center_pos) / 6 # grasp at 1/3 of the peg
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 + np.tanh(10.0 * (peg_head_pos_at_hole[0] + 0.015)) # (0, 2)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1)
                align_reward_z = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[2])) # (0, 1) 
                
                reward += insertion_reward + align_reward_y + align_reward_z

                peg_axis = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += cos * 2

        return reward
    
@register_env("PegInsertionSideFixed_grasp_2-v0", max_episode_steps=200)
class PegInsertionSideEnv_fixed_grasp_2(PegInsertionSideEnv_fixed):
    
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 7.25 + 1
        else:
            # reaching reward
            gripper_pos = self.tcp.pose.p
            peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
            head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
            grasp_pos = center_pos - (head_pos - center_pos) * ((0.015+self.peg_radius)/self.peg_half_length) # hack a grasp point
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg) and (gripper_to_peg_dist < self.peg_radius * 0.9)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 + np.tanh(10.0 * (peg_head_pos_at_hole[0] + 0.015)) # (0, 2)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1)
                align_reward_z = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[2])) # (0, 1) 
                
                reward += insertion_reward + align_reward_y + align_reward_z

                peg_normal = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_normal = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos = abs(np.dot(hole_normal, peg_normal) / np.linalg.norm(peg_normal) / np.linalg.norm(hole_normal)) # (0, 1)
                reward += cos * 2

        return reward
    
@register_env("PegInsertionSideFixed_axis-v0", max_episode_steps=200)
class PegInsertionSideEnv_fixed_axis(PegInsertionSideEnv_fixed):
    
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 7.25 + 1
        else:
            # reaching reward
            gripper_pos = self.tcp.pose.p
            peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
            head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
            grasp_pos = center_pos - (head_pos - center_pos) * ((0.015+self.peg_radius)/self.peg_half_length) # hack a grasp point
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg) and (gripper_to_peg_dist < self.peg_radius * 0.9)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 + np.tanh(10.0 * (peg_head_pos_at_hole[0] + 0.015)) # (0, 2)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1)
                align_reward_z = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[2])) # (0, 1) 
                
                reward += insertion_reward + align_reward_y + align_reward_z

                peg_axis = self.peg.pose.transform(Pose([1,0,0])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([1,0,0])).p - box_hole_pose.p
                cos_axis = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += cos_axis *2

        return reward


@register_env("PegInsertionSideFixed_2cos-v0", max_episode_steps=200)
class PegInsertionSideEnv_fixed_2cos(PegInsertionSideEnv_fixed):
    
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 7.25 + 1
        else:
            # reaching reward
            gripper_pos = self.tcp.pose.p
            peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
            head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
            grasp_pos = center_pos - (head_pos - center_pos) * ((0.015+self.peg_radius)/self.peg_half_length) # hack a grasp point
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg) and (gripper_to_peg_dist < self.peg_radius * 0.9)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 + np.tanh(5.0 * (peg_head_pos_at_hole[0] + 0.015)) # (0, 2)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1)
                align_reward_z = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[2])) # (0, 1) 
                
                reward += insertion_reward + align_reward_y + align_reward_z

                peg_normal = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_normal = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos_normal = abs(np.dot(hole_normal, peg_normal) / np.linalg.norm(peg_normal) / np.linalg.norm(hole_normal)) # (0, 1)
                reward += cos_normal

                peg_axis = self.peg.pose.transform(Pose([1,0,0])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([1,0,0])).p - box_hole_pose.p
                cos_axis = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += cos_axis

        return reward

@register_env("PegInsertionSideFixed_deep-v0", max_episode_steps=200)
class PegInsertionSideEnv_fixed_deep(PegInsertionSideEnv_fixed):
    
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 7.25 + 1
        else:
            # reaching reward
            gripper_pos = self.tcp.pose.p
            peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
            head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
            grasp_pos = center_pos - (head_pos - center_pos) * ((0.015+self.peg_radius)/self.peg_half_length) # hack a grasp point
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg) and (gripper_to_peg_dist < self.peg_radius * 0.9)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 - np.tanh(5.0 * abs(self.peg_half_length - peg_head_pos_at_hole[0])) # (0, 1)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1)
                align_reward_z = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[2])) # (0, 1) 
                
                reward += insertion_reward * 2 + align_reward_y + align_reward_z

                peg_normal = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_normal = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos_normal = abs(np.dot(hole_normal, peg_normal) / np.linalg.norm(peg_normal) / np.linalg.norm(hole_normal)) # (0, 1)
                reward += cos_normal

                peg_axis = self.peg.pose.transform(Pose([1,0,0])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([1,0,0])).p - box_hole_pose.p
                cos_axis = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += cos_axis

        return reward


@register_env("PegInsertionSide-v1", max_episode_steps=200)
class PegInsertionSideEnv_v1(PegInsertionSideEnv):
    
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 7.25 + 1
        else:
            # reaching reward
            gripper_pos = self.tcp.pose.p
            peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
            head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
            grasp_pos = center_pos - (head_pos - center_pos) * ((0.015+self.peg_radius)/self.peg_half_length) # hack a grasp point
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg) and (gripper_to_peg_dist < self.peg_radius * 0.9)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 - np.tanh(5.0 * abs(self.peg_half_length - peg_head_pos_at_hole[0])) # (0, 1)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1)
                align_reward_z = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[2])) # (0, 1) 
                
                reward += insertion_reward * 2 + align_reward_y + align_reward_z

                peg_normal = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_normal = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos_normal = abs(np.dot(hole_normal, peg_normal) / np.linalg.norm(peg_normal) / np.linalg.norm(hole_normal)) # (0, 1)
                reward += cos_normal

                peg_axis = self.peg.pose.transform(Pose([1,0,0])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([1,0,0])).p - box_hole_pose.p
                cos_axis = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += cos_axis

        return reward
    
@register_env("PegInsertionSide-v2", max_episode_steps=200)
class PegInsertionSideEnv_v2(PegInsertionSideEnv_v1):
    # obs with is_grasp
    def _get_obs_extra(self) -> OrderedDict:
        ret = super()._get_obs_extra()
        ret['is_grasped'] = float(self.agent.check_grasp(self.peg))
        return ret
    
@register_env("PegInsertionSide-v3", max_episode_steps=200)
class PegInsertionSideEnv_v3(PegInsertionSideEnv_v2):
    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        # peg
        length, radius = 0.1, 0.02
        self.peg_half_length = length
        self.peg_radius = radius
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[length, radius, radius])

        # peg head
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EC7357"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        # peg tail
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EDF6F9"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([-length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        self.peg = builder.build("peg")
        self.peg_head_offset = Pose([length, 0, 0])
        self.peg_half_size = np.float32([length, radius, radius])

        # box with hole
        center = np.zeros(2)
        inner_radius, outer_radius, depth = radius + self._clearance, length, length
        self.box = self._build_box_with_hole(
            inner_radius, outer_radius, depth, center=center
        )
        self.box_hole_offset = Pose(np.hstack([0, center]))
        self.box_hole_radius = inner_radius

    def _initialize_actors(self):
        xy = np.array([0, -0.15])
        pos = np.hstack([xy, self.peg_half_size[2]])
        ori = np.pi / 2
        quat = euler2quat(0, 0, ori)
        self.peg.set_pose(Pose(pos, quat))

        xy = np.array([0, 0.3])
        pos = np.hstack([xy, self.peg_half_size[0]])
        ori = np.pi / 2
        quat = euler2quat(0, 0, ori)
        self.box.set_pose(Pose(pos, quat))

@register_env("PegInsertionSide-v4", max_episode_steps=200)
class PegInsertionSideEnv_v4(PegInsertionSideEnv_v3):
    def _get_obs_agent(self):
        """Remove gripper's vel and base pose."""
        proprioception = self.agent.get_proprioception()
        proprioception['qvel'] = proprioception['qvel'][:-2]
        return proprioception

@register_env("PegInsertionSide2D-v0", max_episode_steps=200)
class PegInsertionSide2DEnv_v0(PegInsertionSideEnv_v4):
    _clearance = 0.008

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        # peg
        length, radius = 0.075, 0.025
        self.peg_half_length = length
        self.peg_radius = radius
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[length, radius, radius])

        # peg head
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EC7357"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        # peg tail
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EDF6F9"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([-length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        self.peg = builder.build("peg")
        self.peg_head_offset = Pose([length, 0, 0])
        self.peg_half_size = np.float32([length, radius, radius])

        # box with hole
        center = np.zeros(2)
        inner_radius, outer_radius, depth = radius + self._clearance, length, length / 2
        self.box = self._build_box_with_hole(
            inner_radius, outer_radius, depth, center=center
        )
        self.box_hole_offset = Pose(np.hstack([0, center]))
        self.box_hole_radius = inner_radius

    def _initialize_actors(self):
        xy = np.array([-0.1, -0.2])
        pos = np.hstack([xy, self.peg_half_size[2]])
        ori = np.pi
        quat = euler2quat(0, 0, ori)
        self.peg.set_pose(Pose(pos, quat))
        
        xy = np.array([-0.3, -0.2])
        pos = np.hstack([xy, self.peg_half_size[2]])
        ori = np.pi
        quat = euler2quat(0, 0, ori)
        self.box.set_pose(Pose(pos, quat))

    def _build_box_with_hole(
        self, inner_radius, outer_radius, depth, center=(0, 0), name="box_with_hole"
    ):
        builder = self._scene.create_actor_builder()
        thickness = (outer_radius - inner_radius) * 0.5
        # x-axis is hole direction
        half_sizes = [
            [depth, thickness, outer_radius],
            [depth, thickness, outer_radius],
            [depth, outer_radius, thickness],
            # [depth, outer_radius, thickness],
        ]
        offset = thickness + inner_radius
        poses = [
            Pose([0, offset, 0]),
            Pose([0, -offset, 0]),
            Pose([0, 0, offset]),
            # Pose([0, 0, -offset]),
        ]

        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#FFD289"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5

        for (half_size, pose) in zip(half_sizes, poses):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)
        return builder.build_static(name)
    
    def _initialize_agent(self):
        if  self.robot_uid in ['xarm7', 'xarm7_d435']:
            qpos = np.array(
                    [0.0, 0.0, 0.0, np.pi / 6, 0.0, np.pi / 6, 0.0, 0.0446430, 0.0446430]
                )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.4639, 0.0, 0.0]))
        else:
            raise NotImplementedError(self.robot_uid)
    
    def _get_obs_extra(self) -> OrderedDict:
        """
            Delete box_hole_radius and peg_half_size.
        """
        ret = super()._get_obs_extra()
        del ret['box_hole_radius']
        del ret['peg_half_size']
        
        return ret

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
        box_hole_pose = self.box_hole_pose
        peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[0]
        y_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[1] <= self.box_hole_radius
        )
        return (x_flag and y_flag), peg_head_pos_at_hole

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 6.25 + 1
        else:
            # reaching reward
            gripper_pos = self.tcp.pose.p
            peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
            head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
            grasp_pos = center_pos - (head_pos - center_pos) * ((0.015+self.peg_radius)/self.peg_half_length) # hack a grasp point
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg) and (gripper_to_peg_dist < self.peg_radius * 0.9)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 - np.tanh(5.0 * abs(self.peg_half_length - peg_head_pos_at_hole[0])) # (0, 1)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1) 
                
                reward += insertion_reward * 2 + align_reward_y

                peg_normal = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_normal = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos_normal = abs(np.dot(hole_normal, peg_normal) / np.linalg.norm(peg_normal) / np.linalg.norm(hole_normal)) # (0, 1)
                reward += cos_normal

                peg_axis = self.peg.pose.transform(Pose([1,0,0])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([1,0,0])).p - box_hole_pose.p
                cos_axis = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += cos_axis

        return reward
    
@register_env("PegInsertionSide2D-v1", max_episode_steps=200)
class PegInsertionSide2DEnv_v1(PegInsertionSide2DEnv_v0):
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward = 7.25 + 1
        else:
            tcp_at_peg = (self.peg.pose.inv() * self.tcp.pose).p
            y_dist = np.abs(tcp_at_peg[1])
            if y_dist > 0.01:
                # reaching reward 0
                gripper_pos = self.tcp.pose.p
                peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
                head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
                grasp_prepos = center_pos - (head_pos - center_pos) * 1.5 # hack a grasp point
                gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_prepos)
                reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
                reward += reaching_reward
            else:
                # reaching reward 1
                reward += 1
                gripper_pos = self.tcp.pose.p
                peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
                head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
                grasp_pos = center_pos - (head_pos - center_pos) * ((0.015+self.peg_radius)/self.peg_half_length) # hack a grasp point
                gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
                reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
                reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg) and (gripper_to_peg_dist < self.peg_radius * 0.9)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 - np.tanh(5.0 * abs(self.peg_half_length - peg_head_pos_at_hole[0])) # (0, 1)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1) 
                
                reward += insertion_reward * 2 + align_reward_y

                peg_normal = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_normal = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos_normal = abs(np.dot(hole_normal, peg_normal) / np.linalg.norm(peg_normal) / np.linalg.norm(hole_normal)) # (0, 1)
                reward += cos_normal

                peg_axis = self.peg.pose.transform(Pose([1,0,0])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([1,0,0])).p - box_hole_pose.p
                cos_axis = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += cos_axis

        return reward
    
@register_env("PegInsertionSide2D-v2", max_episode_steps=200)
class PegInsertionSide2DEnv_v2(PegInsertionSide2DEnv_v1):
    def _initialize_actors(self):
        xy = self._episode_rng.uniform([-0.2, -0.25], [-0.2, -0.15])
        pos = np.hstack([xy, self.peg_half_size[2]])
        # ori = np.pi + self._episode_rng.uniform(-np.pi / 6, np.pi / 6)
        ori = np.pi
        quat = euler2quat(0, 0, ori)
        self.peg.set_pose(Pose(pos, quat))
        
        xy = np.array([-0.4, -0.2])
        pos = np.hstack([xy, self.peg_half_size[2]])
        ori = np.pi
        quat = euler2quat(0, 0, ori)
        self.box.set_pose(Pose(pos, quat))
    
    def evaluate(self, **kwargs) -> dict:
        is_grasped = self.agent.check_grasp(self.peg)
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        success = success and is_grasped
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)
    
@register_env("PegInsertionSide2D-v3", max_episode_steps=200)
class PegInsertionSide2DEnv_v3(PegInsertionSide2DEnv_v2):
    # def __init__(self, *args, robot="xarm7_d435", robot_init_qpos_noise=0.02, 
    #              domain_rand_params=None, **kwargs):
    #     if domain_rand_params is not None:
    #         self.domain_rand = True
    #         self.size_range = domain_rand_params['size_range']
    #         self.obs_noise = domain_rand_params['obs_noise']
    #     else:
    #         self.domain_rand = False
    #     super().__init__(*args, robot=robot, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        # peg
        length, radius = 0.075, 0.025
        self.peg_half_length = length
        self.peg_radius = radius
        builder = self._scene.create_actor_builder()
        mat = self._scene.create_physical_material(static_friction=0.1, dynamic_friction=0.1, restitution=0.0)
        builder.add_box_collision(half_size=[length, radius, radius], material=mat)

        # peg head
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EC7357"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        # peg tail
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EDF6F9"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([-length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        self.peg = builder.build("peg")
        self.peg_head_offset = Pose([length, 0, 0])
        self.peg_half_size = np.float32([length, radius, radius])

        # box with hole
        center = np.zeros(2)
        inner_radius, outer_radius, depth = radius + self._clearance, length, length / 2
        self.box = self._build_box_with_hole(
            inner_radius, outer_radius, depth, center=center
        )
        self.box_hole_offset = Pose(np.hstack([0, center]))
        self.box_hole_radius = inner_radius

    def compute_dense_reward(self, info, **kwargs):
        reward = super().compute_dense_reward(info, **kwargs)
        if info["time_out"]:
            reward -= 0.0
        return reward
    
@register_env("PegInsertionSide2D-v4", max_episode_steps=200)
class PegInsertionSide2DEnv_v4(PegInsertionSide2DEnv_v3):
    def __init__(self, *args, robot="xarm7_d435", robot_init_qpos_noise=0.02, 
                 domain_rand_params=dict(obs_noise=0.005, joint_noise=0.01, tcp_noise=0.003), **kwargs):
        if domain_rand_params is not None:
            self.domain_rand = True
            self.obs_noise = domain_rand_params['obs_noise']
            self.joint_noise = domain_rand_params['joint_noise']
            self.tcp_noise = domain_rand_params['tcp_noise']
        else:
            self.domain_rand = False
        super().__init__(*args, robot=robot, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    def _get_obs_extra(self) -> OrderedDict:
        """
            Add observation noise to peg.
        """
        ret = super()._get_obs_extra()
        if self.domain_rand:
            xy_noise = self.generate_noise(scale=self.obs_noise, size=2)
            ret['peg_pose'][0:2] += xy_noise
            tcp_noise = self.generate_noise(scale=self.tcp_noise, size=2)
            ret['tcp_pose'][0:2] += tcp_noise

        return ret
    
    def _get_obs_agent(self):
        ret =  super()._get_obs_agent()
        if self.domain_rand:
            ret['qpos'][0:7] += self.generate_noise(scale=self.joint_noise, size=7)
        
        return ret
    
    def generate_noise(self, scale, size):
        noise = np.random.uniform(-scale, scale, size=size)
        return noise
    
@register_env("PegInsertionSide2D-v5", max_episode_steps=100)
class PegInsertionSide2DEnv_v5(PegInsertionSide2DEnv_v4):
    _clearance = 0.006

    def __init__(self, *args, robot="xarm7_d435", robot_init_qpos_noise=0.02, domain_rand_params=dict(obs_noise=0.005, joint_noise=0.01, tcp_noise=0.003), **kwargs):
        super().__init__(*args, robot=robot, robot_init_qpos_noise=robot_init_qpos_noise, domain_rand_params=domain_rand_params, **kwargs)

        self.domain_rand = False

    def _initialize_actors(self):
        xy = self._episode_rng.uniform([-0.15, -0.3], [-0.25, -0.1])
        pos = np.hstack([xy, self.peg_half_size[2]])
        # ori = np.pi + self._episode_rng.uniform(-np.pi / 6, np.pi / 6)
        ori = np.pi
        quat = euler2quat(0, 0, ori)
        self.peg.set_pose(Pose(pos, quat))
        
        xy = np.array([-0.4, -0.2])
        pos = np.hstack([xy, self.peg_half_size[2]])
        ori = np.pi
        quat = euler2quat(0, 0, ori)
        self.box.set_pose(Pose(pos, quat))