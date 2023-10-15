import numpy as np
from sapien.core import Pose

from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2quat, quat2euler


class PickCubeV3HandcraftPolicy:
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
    
    def _to_tcp_frame(self, action, quat=None):
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
        while not done:
            if delta_euler[2] > np.pi / 4:
                delta_euler[2] -= np.pi / 2
                continue
            elif delta_euler[2] < -np.pi / 4:
                delta_euler[2] += np.pi / 2
                continue
            else:
                done = True

        return delta_euler
        

    def predict(self, obs):
        tcp_pose = self._to_sapien_pose(obs[34:41])
        obj_pose = self._to_sapien_pose(obs[47:54])
        tcp_to_obj_pos = obs[54:57]
        obj_to_goal_pos = obs[57:60]
        obj_grasped = obs[60:61]    
        if self.action_step == 0:
            obj_grasped = 0.0

        delta_quat = (obj_pose.inv() * self.dummy_pose).q
        delta_euler = self._nomalize_delta_euler(list(quat2euler(delta_quat)))
        if self.determine_rotate == False:
            self.rotate_angles = delta_euler[2]
            self.rotate_times = int(np.abs(self.rotate_angles) / 0.1) + 1
            self.rotate_direction = 1.0 if self.rotate_angles > 0 else -1.0
            self.determine_rotate = True
        
        action = np.zeros([7])
        action[-1] = -0.7
        if obj_grasped == 0.0:      # obs isn't grasped yet
            self.first_target = tcp_to_obj_pos + np.array([0.0, 0.0, 0.05])
            if (np.linalg.norm(self.first_target) > 0.005 or self.action_step < self.rotate_times) and self.stage==0:
                action[0:3] = np.clip(self._to_tcp_frame(self.first_target, tcp_pose.q) * 10, -1, 1)
                if self.action_step < self.rotate_times - 1:
                    action[-2] = self.rotate_direction
                elif self.action_step == self.rotate_times - 1:
                    action[-2] = (np.abs(self.rotate_angles) - (self.rotate_times - 1) * 0.1) * self.rotate_direction
                action[-1] = 1.0
            elif np.linalg.norm(tcp_to_obj_pos) > 0.005:
                self.stage = 1
                action[0:3] = np.clip(self._to_tcp_frame(tcp_to_obj_pos, tcp_pose.q) * 10, -0.5, 0.5)
                action[-1] = 1.0
            else:
                self.stage = 2
                # if np.linalg.norm(obj_to_goal_pos) > 0.005:
                action[0:3] = np.clip(self._to_tcp_frame(obj_to_goal_pos, tcp_pose.q) * 10, -0.5, 0.5)
            # elif self.action_step == 2:
            #     action[-1] = -0.5
        else:
            # if np.linalg.norm(obj_to_goal_pos) > 0.005:
            action[0:3] = np.clip(self._to_tcp_frame(obj_to_goal_pos, tcp_pose.q) * 10, -0.5, 0.5)


        self.action_step += 1

        return action