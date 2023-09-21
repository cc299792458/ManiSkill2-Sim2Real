import numpy as np
import sapien.core as sapien

class PartialKinematicModel:
    def __init__(self, robot: sapien.Articulation, start_joint_name: str, end_joint_name: str):
        self.original_robot = robot
        self.start_joint_tuple = \
            [(joint, num) for num, joint in enumerate(robot.get_joints()) if
             joint.get_name() == start_joint_name][0]
        self.end_joint_tuple = \
            [(joint, num) for num, joint in enumerate(robot.get_joints()) if joint.get_name() == end_joint_name][
                0]
        self.start_link = self.start_joint_tuple[0].get_parent_link()
        self.end_link = self.end_joint_tuple[0].get_child_link()

        # Build new articulation for partial kinematics chain
        scene = robot.get_builder().get_scene()
        builder = scene.create_articulation_builder()
        root = builder.create_link_builder()
        root.set_mass_and_inertia(
            self.start_link.get_mass(),
            self.start_link.cmass_local_pose,
            self.start_link.get_inertia(),
        )
        links = [root]
        all_joints = robot.get_joints()[self.start_joint_tuple[1]: self.end_joint_tuple[1] + 1]
        for j_idx, j in enumerate(all_joints):
            link = builder.create_link_builder(links[-1])
            link.set_mass_and_inertia(
                j.get_child_link().get_mass(),
                j.get_child_link().cmass_local_pose,
                j.get_child_link().get_inertia(),
            )
            link.set_joint_properties(
                j.type, j.get_limits(), j.get_pose_in_parent(), j.get_pose_in_child()
            )
            link.set_name(j.get_child_link().get_name())
            links.append(link)

        partial_robot = builder.build(fix_root_link=True)
        partial_robot.set_pose(sapien.Pose([0, 0, -10]))
        self.model = partial_robot.create_pinocchio_model()

        # Parse new model
        self.dof = partial_robot.dof
        self.end_link_name = self.end_link.get_name()
        self.end_link_index = [i for i, link in enumerate(partial_robot.get_links()) if
                               link.get_name() == self.end_link_name][0]
        self.partial_robot = partial_robot

    def compute_end_link_spatial_jacobian(self, partial_qpos):
        self.partial_robot.set_qpos(partial_qpos)
        jacobian = self.partial_robot.compute_world_cartesian_jacobian()[
                   self.end_link_index * 6 - 6: self.end_link_index * 6, :]
        return jacobian
    

def compute_inverse_kinematics(delta_pose_world, palm_jacobian, damping=0.05):
    lmbda = np.eye(6) * (damping ** 2)
    # When you need the pinv for matrix multiplication, always use np.linalg.solve but not np.linalg.pinv
    delta_qpos = palm_jacobian.T @ \
                 np.linalg.lstsq(palm_jacobian.dot(palm_jacobian.T) + lmbda, delta_pose_world, rcond=None)[0]

    return delta_qpos