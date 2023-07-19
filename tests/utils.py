import numpy as np
import torch

from mani_skill2.utils.common import flatten_dict_keys
ENV_IDS = [
    "LiftCube-v0",
    "PickCube-v0",
    "StackCube-v0",
    "PickSingleYCB-v0",
    "PickSingleEGAD-v0",
    "PickClutterYCB-v0",
    "AssemblingKits-v0",
    "PegInsertionSide-v0",
    "PlugCharger-v0",
    "PandaAvoidObstacles-v0",
    "TurnFaucet-v0",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]
OBS_MODES = [
    "state_dict",
    "state",
    "rgbd",
    "pointcloud",
    "rgbd_robot_seg",
    "pointcloud_robot_seg",
]
ROBOTS = [
    "panda", "xmate3_robotiq"
]
def assert_obs_equal(obs1, obs2):
    if isinstance(obs1, dict):
        obs2 = flatten_dict_keys(obs2)
        for k, v in obs1.items():
            v2 = obs2[k]
            if isinstance(v2, torch.Tensor):
                v2 = v2.cpu().numpy()
            if v.dtype == np.uint8:
                # https://github.com/numpy/numpy/issues/19183
                np.testing.assert_allclose(
                    np.float32(v), np.float32(v2), err_msg=k, atol=1
                )
            elif np.issubdtype(v.dtype, np.integer):
                np.testing.assert_equal(v, v2, err_msg=k)
            else:
                np.testing.assert_allclose(v, v2, err_msg=k, atol=1e-4)
    else:
        np.testing.assert_allclose(obs1, obs2, atol=1e-4)