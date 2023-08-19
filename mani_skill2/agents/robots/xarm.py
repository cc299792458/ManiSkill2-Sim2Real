import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.configs.xarm import defaults
from mani_skill2.utils.common import compute_angle_between
from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
)


class XArm(BaseAgent):
    _config: defaults.XArmDefaultConfig

    @classmethod
    def get_default_config(cls):
        raise NotImplementedError()

    def _after_init(self):
        self.finger1_link: sapien.LinkBase = get_entity_by_name(
            self.robot.get_links(), "left_finger"
        )
        self.finger2_link: sapien.LinkBase = get_entity_by_name(
            self.robot.get_links(), "right_finger"
        )

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[:3, 1]

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

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose.from_transformation_matrix(T)


class XArm7(XArm):
    _config: defaults.XArm7DefaultConfig

    @classmethod
    def get_default_config(cls, sim_params):
        return defaults.XArm7DefaultConfig(sim_params)


class XArm7D435(XArm):
    _config: defaults.XArm7D435DefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.XArm7D435DefaultConfig()
