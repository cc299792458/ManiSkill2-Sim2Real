import numpy as np
from sapien.core import Pose

from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2quat, quat2euler


class PegInsertionSideV2HandcraftPolicy:
    def __init__(self):
        self.action_step = 0
        self.obj_ori_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.dummy_pose = Pose()
        self.dummy_pose.set_q(self.obj_ori_quat)
        self.rotate_times = 0
        self.rotate_angles = 0
        self.determine_rotate = False
        self.stage = 0

    def reset(self):
        self.action_step = 0
        self.obj_ori_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.dummy_pose = Pose()
        self.dummy_pose.set_q(self.obj_ori_quat)
        self.rotate_times = 0
        self.rotate_angles = 0
        self.determine_rotate = False
        self.stage = 0

    def _to_sapien_pose(self, pose):
        sapien_pose = Pose()
        sapien_pose.set_p(pose[0:3])
        sapien_pose.set_q(pose[3:7])

        return sapien_pose
    
    def _to_frame(self, action, quat=None):
        # if action.shape[0] == 3:
        #     convert_action = np.zeros([3])
        #     convert_action[0] = action[1]
        #     convert_action[1] = action[0]
        #     convert_action[2] = -action[2]

        # if quat is not None:
        w, x, y, z = quat
        rot_mat = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
                            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]])

        convert_action = rot_mat @ action

        return convert_action
        
    def _nomalize_delta_euler(self, delta_euler):
        done = False
        delta_euler[2] += np.pi / 2
        while not done:
            if delta_euler[2] > np.pi / 2:
                delta_euler[2] -= np.pi
                continue
            elif delta_euler[2] < -np.pi / 2:
                delta_euler[2] += np.pi
                continue
            else:
                done = True

        return delta_euler
        

    def predict(self, obs):
        tcp_pose = self._to_sapien_pose(obs[34:41])
        peg_pose = self._to_sapien_pose(obs[41:48])
        box_hole_pose = self._to_sapien_pose(obs[48:55])
        tcp_to_peg_pos = obs[55:58]
        peg_to_box_hole_pos = obs[58:61]
        peg_is_grasped = obs[61]
        
        action = np.zeros([7])
        action[-1] = -0.7
        if self.action_step == 0:
            peg_is_grasped = 0.0
        delta_quat = (peg_pose.inv() * self.dummy_pose).q
        delta_euler = self._nomalize_delta_euler(list(quat2euler(delta_quat)))
        if self.determine_rotate == False:
            self.rotate_angles = delta_euler[2]
            self.rotate_times = int(np.abs(self.rotate_angles) / 0.1) + 1
            self.rotate_direction = 1.0 if self.rotate_angles > 0 else -1.0
            self.determine_rotate = True
        

        if peg_is_grasped == 0.0:
            z_rot = Pose()
            z_rot.set_q(np.array([0.707, 0.0, 0.0, 0.707]))
            target_pose = peg_pose * z_rot
            delta_x = self._to_frame([0.0, 0.05, 0.0], target_pose.inv().q)
            delta_x[0] = -delta_x[0]
            self.first_target = tcp_to_peg_pos + np.array([0.0, 0.0, 0.05]) + delta_x
            self.second_target = tcp_to_peg_pos + delta_x 
            if (np.linalg.norm(self.first_target) > 0.005 or self.action_step < self.rotate_times) and self.stage==0:
                action[0:3] = np.clip(self._to_frame(self.first_target, tcp_pose.q) * 10, -1, 1)
                action[-1] = 1.0
                if self.action_step < self.rotate_times - 1:
                    action[-2] = self.rotate_direction
                elif self.action_step == self.rotate_times - 1:
                    action[-2] = (np.abs(self.rotate_angles) - (self.rotate_times - 1) * 0.1) * self.rotate_direction
            elif np.linalg.norm(self.second_target) > 0.005:
                self.stage = 1
                action[0:3] = np.clip(self._to_frame(self.second_target, tcp_pose.q) * 10, -0.5, 0.5)
                action[-1] = 1.0
            else:
                self.stage = 2
                if np.linalg.norm(peg_to_box_hole_pos[2]) > 0.005:
                    action[2] = np.clip(self._to_frame(peg_to_box_hole_pos, tcp_pose.q) * 10, -0.5, 0.5)[2]
                # action[0:3] = np.clip(self._to_frame(obj_to_goal_pos, tcp_pose.q) * 10, -0.5, 0.5)
        else:
            if np.linalg.norm(peg_to_box_hole_pos[2]) > 0.005:
                action[2] = np.clip(self._to_frame(peg_to_box_hole_pos, tcp_pose.q) * 10, -0.5, 0.5)[2]
            else:
                action[0:3] = np.clip(self._to_frame(peg_to_box_hole_pos, tcp_pose.q) * 10, -0.5, 0.5)
        self.action_step += 1

        return action