from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import sapien.core as sapien
from gym import spaces
from scipy.spatial.transform import Rotation

from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose
from mani_skill2.utils.common import clip_and_scale_action
from mani_skill2.utils.kinematics_helper import PartialKinematicModel, compute_inverse_kinematics

from ..base_controller import BaseController, ControllerConfig

class PDEEVelPosController(BaseController):
    config: "PDEEVelPosControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()

        # hard-coded with xarm7
        self.start_joint_name = self.articulation.get_joints()[1].get_name()
        self.end_joint_name = self.articulation.get_active_joints()[6].get_name()
        self.kinematic_model = PartialKinematicModel(self.articulation, self.start_joint_name, self.end_joint_name)
        self.vel_ee_link_name = self.kinematic_model.end_link_name
        self.vel_ee_link = [link for link in self.articulation.get_links() if link.get_name() == self.vel_ee_link_name][0]

        if self.config.ee_link:
            self.ee_link = get_entity_by_name(
                self.articulation.get_links(), self.config.ee_link
            )
        else:
            # The child link of last joint is assumed to be the end-effector.
            self.ee_link = self.joints[-1].get_child_link()
        self.ee_link_idx = self.articulation.get_links().index(self.ee_link)

    def _initialize_action_space(self):
        low = np.float32(np.broadcast_to(self.config.lower, 3))
        high = np.float32(np.broadcast_to(self.config.upper, 3))
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        super().reset()
        self._step = 0  # counter of simulation steps after action is set
        self._target_qpos = self.qpos
        self._target_qvel = np.zeros_like(self.qpos)

    def set_drive_property(self):
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            joint.set_drive_property(
                stiffness[i], damping[i], force_limit=force_limit[i]
            )
            joint.set_friction(friction[i])

    def _preprocess_action(self, action: np.ndarray):
        return np.hstack([super()._preprocess_action(action), np.zeros([3])])

    def _postprocess_action(self, action: np.ndarray):
        return np.clip(action, -self.config.limit, self.config.limit)
    
    def set_drive_targets(self, targets):
        for i, joint in enumerate(self.joints):
            joint.set_drive_target(targets[i])
    
    def set_drive_velocity_target(self, velocity_target):
        for i, joint in enumerate(self.joints):
            joint.set_drive_velocity_target(velocity_target[i])
    
    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        palm_jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(self.qpos)
        self._target_qvel = self._postprocess_action(compute_inverse_kinematics(action, palm_jacobian))
        self._target_qpos = self.qpos + self._target_qvel * (1 / self.control_freq)
        
        self.set_drive_targets(self._target_qpos)
        self.set_drive_velocity_target(self._target_qvel)



@dataclass
class PDEEVelPosControllerConfig(ControllerConfig):
    lower: Union[float, Sequence[float]]
    upper: Union[float, Sequence[float]]
    limit: Union[float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    ee_link: str = None
    normalize_action: bool = True
    controller_cls = PDEEVelPosController

