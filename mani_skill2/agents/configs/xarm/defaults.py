from copy import deepcopy
import numpy as np

from mani_skill2.agents.controllers import *
from mani_skill2.sensors.camera import CameraConfig


class XArmDefaultConfig:
    def __init__(self) -> None:
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                left_finger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                right_finger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),  
        )
        self.gripper_joint_names = [
            "left_finger_joint",
            "right_finger_joint",
        ]
        self.ee_low, self.ee_high = -0.0005, 0.0446430
        self.ee_cls_cfg = PDJointPosMimicControllerConfig
        self.ee_link_name = "link_tcp"

        self.arm_use_target = False # NOTE(chichu): enable to calculate next_target_pose based on current_target_pose, making it easier for sim2real

    @property
    def controllers(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
            use_target=self.arm_use_target, 
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
            use_target=self.arm_use_target,
        )

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            use_target=self.arm_use_target,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            use_target=self.arm_use_target,
        )
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -np.pi,
            np.pi,
            self.arm_damping,
            self.arm_force_limit,
        )
        arm_pd_ee_vel_pos = PDEEVelPosControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            np.pi/2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
        )
        arm_pd_ee_vel_pose = PDEEVelPoseControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            1.0,
            np.pi/2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = self.ee_cls_cfg(
            self.gripper_joint_names,
            self.ee_low,  # a trick to have force when the object is thin
            self.ee_high,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_target=self.arm_use_target,
        )

        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            arm_pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos
            ),
            pd_ee_vel_pos=dict(
                arm=arm_pd_ee_vel_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_vel_pose=dict(
                arm=arm_pd_ee_vel_pose, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        return CameraConfig(
            uid="hand_camera",
            p=[-0.0464982, 0.0200011, 0.0360011],
            q=[-0.70710678, 0, 0.70710678, 0],
            width=128,
            height=128,
            fov=1.57,
            near=0.01,
            far=10,
            actor_uid="xarm_gripper_base_link",
        )


class XArm7DefaultConfig(XArmDefaultConfig):
    def __init__(self) -> None:
        super().__init__()
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/xarm7_reduced_gripper.urdf"
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]


class XArm7D435DefaultConfig(XArm7DefaultConfig):
    def __init__(self) -> None:
        super().__init__()
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/xarm7_reduced_gripper_d435.urdf"

    @property
    def cameras(self):
        return CameraConfig(
            uid="hand_camera",
            p=[0, 0, 0],
            q=[1, 0, 0, 0],
            width=848,
            height=480,
            fov=np.deg2rad(43.5),
            near=0.01,
            far=10,
            actor_uid="camera_color_frame",
        )