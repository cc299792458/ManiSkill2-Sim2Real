from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np
import sapien.core as sapien
from gym import spaces

from mani_skill2 import format_path
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.sapien_utils import check_urdf_config, parse_urdf_config

from .base_controller import BaseController, CombinedController, ControllerConfig


@dataclass
class AgentConfig:
    """Agent configuration.

    Args:
        urdf_path: path to URDF file. Support placeholders like {PACKAGE_ASSET_DIR}.
        urdf_config: a dict to specify materials and simulation parameters when loading URDF from SAPIEN.
        controllers: a dict of controller configurations
        cameras: a dict of onboard camera configurations
    """

    urdf_path: str
    urdf_config: dict
    controllers: Dict[str, Union[ControllerConfig, Dict[str, ControllerConfig]]]
    cameras: Dict[str, CameraConfig]


class BaseAgent:
    """Base class for agents.

    Agent is an interface of the robot (sapien.Articulation).

    Args:
        scene (sapien.Scene): simulation scene instance.
        control_freq (int): control frequency (Hz).
        control_mode: uid of controller to use
        fix_root_link: whether to fix the robot root link
        config: agent configuration
    """

    robot: sapien.Articulation
    controllers: Dict[str, BaseController]

    def __init__(
        self,
        scene: sapien.Scene,
        control_freq: int,
        control_mode: str = None,
        fix_root_link=True,
        config: AgentConfig = None,
        sim_params: dict = None,
        ee_type: str = None,
        ee_move_independently: bool = None,
    ):
        self.scene = scene
        self._control_freq = control_freq

        self.config = config or self.get_default_config(sim_params)

        # URDF
        self.urdf_path = self.config.urdf_path
        self.fix_root_link = fix_root_link
        self.urdf_config = self.config.urdf_config

        # Controller
        self.controller_configs = self.config.controllers
        self.supported_control_modes = list(self.controller_configs.keys())
        if control_mode is None:
            control_mode = self.supported_control_modes[0]
        # The control mode after reset for consistency
        self._default_control_mode = control_mode

        # EE move independently or not
        self.ee_move_independently = ee_move_independently

        self._load_articulation(ee_type)
        self._setup_controllers()
        self.set_control_mode(control_mode)
        self._after_init()

    @classmethod
    def get_default_config(cls) -> AgentConfig:
        raise NotImplementedError

    def _load_articulation(self, ee_type):
        loader = self.scene.create_urdf_loader()
        urdf_path = format_path(str(self.urdf_path))
        urdf_config = parse_urdf_config(self.urdf_config, self.scene)
        check_urdf_config(urdf_config)
        robot_builder = loader.load_file_as_articulation_builder(urdf_path, urdf_config)
        if ee_type == 'full_gripper':
            link_group = ['right_outer_knuckle', 'right_finger', 'right_inner_knuckle',
                          'left_outer_knuckle', 'left_finger', 'left_inner_knuckle',
                          'xarm_gripper_base_link']
            self._set_collision_group(robot_builder, link_group)
        # TODO(jigu): support loading multiple convex collision shapes
        self.robot = robot_builder.build(fix_root_link=self.fix_root_link)
        assert self.robot is not None, f"Fail to load URDF from {urdf_path}"
        self.robot.set_name(Path(urdf_path).stem)
        if ee_type == 'full_gripper':
            self._add_constraint()
        # Cache robot link ids
        self.robot_link_ids = [link.get_id() for link in self.robot.get_links()]

    def _setup_controllers(self):
        self.controllers = OrderedDict()
        for uid, config in self.controller_configs.items():
            if isinstance(config, dict):
                self.controllers[uid] = CombinedController(
                    config, self.robot, self._control_freq
                )
            else:
                self.controllers[uid] = config.controller_cls(
                    config, self.robot, self._control_freq
                )

    def _after_init(self):
        """After initialization. E.g., caching the end-effector link."""
        pass

    def _add_constraint(self):
        # Add left finger and right finget constraints, respectively
        finger = next(j for j in self.robot.get_active_joints() if j.name == "left_finger_joint")
        inner_knuckle = next(j for j in self.robot.get_active_joints() if j.name == "left_inner_knuckle_joint")
        pad, lif = finger.get_child_link(), inner_knuckle.get_child_link()
        # NOTE(chichu): p_f and p_p are calculated in advance
        p_f = np.array([-1.7706577e-07, 3.5465002e-02, 4.2038992e-02], dtype=np.float32)
        p_p = np.array([-1.6365516e-07, -1.4999986e-02, 1.4999956e-02], dtype=np.float32) 
        left_drive = self.scene.create_drive(lif, sapien.Pose(p_f), pad, sapien.Pose(p_p))
        left_drive.lock_motion(1, 1, 1, 0, 0, 0)

        finger = next(j for j in self.robot.get_active_joints() if j.name == "right_finger_joint")
        inner_knuckle = next(j for j in self.robot.get_active_joints() if j.name == "right_inner_knuckle_joint")
        pad, lif = finger.get_child_link(), inner_knuckle.get_child_link()
        p_f = np.array([-7.7380150e-08, -3.5464913e-02, 4.2038962e-02], dtype=np.float32)
        p_p = np.array([-9.3990820e-08, 1.5000075e-02, 1.4999941e-02], dtype=np.float32)
        right_drive = self.scene.create_drive(lif, sapien.Pose(p_f), pad, sapien.Pose(p_p))
        right_drive.lock_motion(1, 1, 1, 0, 0, 0)

    def _set_collision_group(self, robot_builder, link_group):
        for link_builder in robot_builder.get_link_builders():
            if link_builder.get_name() in link_group:
                link_builder.set_collision_groups(1, 1, 2, 0)

    @property
    def control_mode(self):
        """Get the currently activated controller uid."""
        return self._control_mode

    def set_control_mode(self, control_mode):
        """Set the controller and reset."""
        assert (
            control_mode in self.supported_control_modes
        ), "{} not in supported modes: {}".format(
            control_mode, self.supported_control_modes
        )
        self._control_mode = control_mode
        self.controller.reset()

    @property
    def controller(self):
        """Get currently activated controller."""
        if self._control_mode is None:
            raise RuntimeError("Please specify a control mode first")
        else:
            return self.controllers[self._control_mode]

    @property
    def action_space(self):
        if self._control_mode is None:
            return spaces.Dict(
                {
                    uid: controller.action_space
                    for uid, controller in self.controllers.items()
                }
            )
        else:
            if self.ee_move_independently == False:
                return self.controller.action_space
            else:
                # NOTE(chichu): extra dim is used to judge if ee moves or arm moves
                original_action_space: spaces.Box = self.controller.action_space
                low = original_action_space.low[0]
                high = original_action_space.high[0]
                dim = original_action_space.shape[0]
                return spaces.Box(low=low, high=high, shape=(dim+1,))

    def reset(self, init_qpos=None):
        if init_qpos is not None:
            self.robot.set_qpos(init_qpos)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.robot.set_qacc(np.zeros(self.robot.dof))
        self.robot.set_qf(np.zeros(self.robot.dof))
        # Add set initial qpos drive here.
        self.robot.set_drive_target(init_qpos)
        self.robot.set_drive_velocity_target(np.zeros_like(init_qpos))
        self.set_control_mode(self._default_control_mode)

    def set_action(self, action):
        self.controller.set_action(action, self.ee_move_independently)

    def get_target_qpos(self):
        return self.controller.get_target_qpos()
    
    def get_target_ee_pose(self):
        return self.controller.get_target_ee_pose()
    
    def get_ee_pose(self):
        return self.controller.get_ee_pose()

    def before_simulation_step(self, only_ee=False):
        self.controller.before_simulation_step(only_ee=only_ee)

    # -------------------------------------------------------------------------- #
    # Observations
    # -------------------------------------------------------------------------- #
    def get_proprioception(self):
        obs = OrderedDict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
        controller_state = self.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def get_state(self) -> Dict:
        """Get current state for MPC, including robot state and controller state"""
        state = OrderedDict()

        # robot state
        root_link = self.robot.get_links()[0]
        state["robot_root_pose"] = root_link.get_pose()
        state["robot_root_vel"] = root_link.get_velocity()
        state["robot_root_qvel"] = root_link.get_angular_velocity()
        state["robot_qpos"] = self.robot.get_qpos()
        state["robot_qvel"] = self.robot.get_qvel()
        state["robot_qacc"] = self.robot.get_qacc()

        # controller state
        state["controller"] = self.controller.get_state()

        return state

    def set_state(self, state: Dict, ignore_controller=False):
        # robot state
        self.robot.set_root_pose(state["robot_root_pose"])
        self.robot.set_root_velocity(state["robot_root_vel"])
        self.robot.set_root_angular_velocity(state["robot_root_qvel"])
        self.robot.set_qpos(state["robot_qpos"])
        self.robot.set_qvel(state["robot_qvel"])
        self.robot.set_qacc(state["robot_qacc"])

        if not ignore_controller and "controller" in state:
            self.controller.set_state(state["controller"])
