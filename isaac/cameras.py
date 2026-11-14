"""Two Eye cameras for PackingSim (Isaac Lab 2.3 CameraCfg API from memory; VERIFY on first run).
Needs the app launched with --enable_cameras."""
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaac.packing_sim import PackSceneCfg


def _cam(name):
    return CameraCfg(prim_path="{ENV_REGEX_NS}/" + name, update_period=0.0, height=480, width=640, data_types=["rgb"],
                     spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                                      clipping_range=(0.05, 6.0)),
                     offset=CameraCfg.OffsetCfg(pos=(0.0, -1.0, 0.8), rot=(1.0, 0.0, 0.0, 0.0), convention="world"))


@configclass
class PackSceneWithCamsCfg(PackSceneCfg):
    camera_A = _cam("CamA")
    camera_B = _cam("CamB")


# eye position, look-at target (world). Different viewpoints so each leaves part of the table occluded/oblique.
EYE_POSES = {"A": ((-0.75, -0.75, 0.75), (0.0, 0.15, 0.0)), "B": ((0.75, -0.75, 0.75), (0.0, 0.15, 0.0))}
