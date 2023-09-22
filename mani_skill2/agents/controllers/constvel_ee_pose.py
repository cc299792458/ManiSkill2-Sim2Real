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

class ConstVelEEPoseController(PDEEPoseController):
    config: "ConstVelEEPoseControllerConfig"

    def __init__(self, config: ControllerConfig, articulation, control_freq: int, sim_freq: int = None, 
                 trans_vel=0.2, rot_vel=0.314, interpolate_step=5): 
        """
            This controller tries to track a constant linear motion trajectory.
            
            Args:
                trans_vel: velocity of constant translate linear motion.
                rot_vel: angular velocity of constant rotate linear motion.
                interpolate_step: how many simulation steps to set a sub target.
            
            Real XArm has translation velocity 0.1 m/s, rotation velcocity 0.314 rad/s.
                    
            Single sub target's distance can be calculated by: trans_vel * interpolate_step * (1 / simulation_freq)
            Since simulation_freq is fixed to 200Hz, the distance equals to trans_vel * interpolate_step / 200.
            If trans_vel=0.1, interpolate_step=25, then single sub-target's distance equals to 0.0125m.
        """
        
        super().__init__(config, articulation, control_freq, sim_freq)
        self.trans_vel = trans_vel
        self.rot_vel = rot_vel
        self.interpolate_step = interpolate_step
        self._initialize_velocity_ik()

    def _initialize_velocity_ik(self):
        # NOTE(chichu): Hard-coded with xarm
        self.start_joint_name = self.articulation.get_joints()[1].get_name()
        self.end_joint_name = self.articulation.get_active_joints()[6].get_name()
        self.kinematic_model = PartialKinematicModel(self.articulation, self.start_joint_name, self.end_joint_name)
        self.vel_ee_link_name = self.kinematic_model.end_link_name
        self.vel_ee_link = [link for link in self.articulation.get_links() if link.get_name() == self.vel_ee_link_name][0]

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        self._step = 0
        self._sub_target_step = 0
        self._start_qpos = self.qpos
        
        # Calculate next target pose/qpos, namely, final target pose/qpos for this action.
        if self.config.use_target:
            prev_ee_pose_at_base = self._target_pose
        else:
            prev_ee_pose_at_base = self.ee_pose_at_base
        self._target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
        self._target_qpos = self.compute_ik(self._target_pose)
        if self._target_qpos is None:
            self._target_qpos = self._start_qpos

        if self.config.interpolate:
            # NOTE(chichu): When we enable use_target, we use last target pose to calculate next target pose,
            # but current pose is used to calculate pos_dis and self._sim_steps.
            pos_dis = np.linalg.norm(self._target_pose.p - self.ee_pose_at_base.p)
            self._sim_steps = pos_dis / (self.trans_vel * (1 / self._sim_freq))
            self._velocity = np.zeros([6])
            self._velocity[0:3] = self.trans_vel * (self._target_pose.p - self.ee_pose_at_base.p) / pos_dis 
            # Calculate sub targets
            self._sub_target_num = int(self._sim_steps // self.interpolate_step) + 1
            self._sub_target_pose = []
            for i in range(self._sub_target_num - 1):
                self._sub_target_pose.append(sapien.Pose(\
                    p=(self.ee_pose_at_base.p + (i + 1) * self._velocity[0:3] * (1 / self._sim_freq * self.interpolate_step)),\
                    q=self._target_pose.q))
            self._sub_target_pose.append(sapien.Pose(p=self._target_pose.p, q=self._target_pose.q))
        else:
            raise NotImplementedError
    
    def compute_velocity_ik(self):
        current_qpos = self.qpos
        jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(current_qpos[:7])
        target_qvel = compute_inverse_kinematics(self._velocity, jacobian)[:7]

        return target_qvel
    
    def before_simulation_step(self):
        """
            Set target in the simulation loop, run certain simulation steps, set target once, depanding on self.interpolate_step.
        """
        if self.config.interpolate:
            if self._step % self.interpolate_step == 0:
                sub_target = self._sub_target_pose[self._sub_target_step]
                self.target_qpos = self.compute_ik(sub_target)
                self.set_drive_targets(self.target_qpos)
                if self._sub_target_step < self._sub_target_num - 1:
                    self.target_qvel = self.compute_velocity_ik() 
                    self.set_drive_velocity_targets(self.target_qvel)
                else:
                    self.set_drive_velocity_targets(np.zeros_like(self.target_qvel))
                if self._sub_target_step < self._sub_target_num - 1:
                    self._sub_target_step += 1
            self._step += 1

        else:
            raise NotImplementedError

@dataclass
class ConstVelEEPoseControllerConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float]]
    pos_upper: Union[float, Sequence[float]]
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
    controller_cls = ConstVelEEPoseController
