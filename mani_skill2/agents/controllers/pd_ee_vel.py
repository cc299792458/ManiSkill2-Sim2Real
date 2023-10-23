from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import sapien.core as sapien
from gym import spaces
from scipy.spatial.transform import Rotation
from mani_skill2.agents.base_controller import ControllerConfig

from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose
from mani_skill2.utils.common import clip_and_scale_action
from mani_skill2.utils.kinematics_helper import PartialKinematicModel, compute_inverse_kinematics

from ..base_controller import BaseController, ControllerConfig
from .pd_ee_pose import PDEEPoseController

import time

class PDEEVelController(PDEEPoseController):
    config: "PDEEVelControllerConfig"

    def __init__(self, config: ControllerConfig, articulation, control_freq: int, sim_freq: int = None): 
        """
            This controller control the velocity of end effector
        """
        
        super().__init__(config, articulation, control_freq, sim_freq)
        self._initialize_velocity_ik()

    def reset(self):
        super().reset()
        self.target_qvel = np.zeros_like(self.qpos)

    def _initialize_velocity_ik(self):
        # NOTE(chichu): Hard-coded with xarm
        self.start_joint_name = self.articulation.get_joints()[1].get_name()
        self.end_joint_name = self.articulation.get_active_joints()[6].get_name()
        self.kinematic_model = PartialKinematicModel(self.articulation, self.start_joint_name, self.end_joint_name)
        self.vel_ee_link_name = self.kinematic_model.end_link_name
        self.vel_ee_link = [link for link in self.articulation.get_links() if link.get_name() == self.vel_ee_link_name][0]

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)
        self._start_qpos = self.qpos
        palm_jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(self._start_qpos)
        self._target_qvel = compute_inverse_kinematics(action, palm_jacobian)
        self._target_qvel = np.clip(self._target_qvel, -np.pi / 1, np.pi / 1)
        self._target_qpos = self._target_qvel * (1 / self.control_freq) + self._start_qpos

        self.set_drive_targets(self._target_qpos)
        self.set_drive_velocity_targets(self._target_qvel)
    
    def before_simulation_step(self):
        self._step += 1

@dataclass
class PDEEVelControllerConfig(ControllerConfig):
    vel_lower: Union[float, Sequence[float]]
    vel_upper: Union[float, Sequence[float]]
    rot_bound: float
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    ee_link: str = None
    frame: str = "ee"  # [base, ee, ee_align]
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    controller_cls = PDEEVelController
