# import sys
# sys.path.append('/home/chichu/Documents/Sapien/ManiSkill2-Sim2Real')

from real_robot.utils.registration import register_env
from mani_skill2.envs.pick_and_place.pick_cube import (
    PickCubeEnv_v2
)
# from .base_env import XArmBaseEnv
from .base_env import XArmBaseEnv

@register_env("PickCubeRealXArm7-v0", max_episode_steps=50)
class PickCubeRealEnv(XArmBaseEnv, PickCubeEnv_v2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)