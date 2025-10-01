from __future__ import annotations
import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import (
    SceneEntityCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    ActionTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg


# Scene
class RobotPickSceneCfg(InteractiveSceneCfg):
    add_ground_plane: bool = True

    robot = ArticulationCfg(
    prim_path="/World/MyRobot",   
    spawn=None                    
)

    cube = RigidObjectCfg(
    prim_path="/World/Cube",      
    spawn=None                   
)


# Config
class RobotPickObsCfg(ObservationGroupCfg):
   joint_pos: ObservationTermCfg = ObservationTermCfg(
    func="omni.isaac.lab.terms.joint_pos",
    params={"asset_cfg": SceneEntityCfg("/World/MyRobot")}
)

joint_vel: ObservationTermCfg = ObservationTermCfg(
    func="omni.isaac.lab.terms.joint_vel",
    params={"asset_cfg": SceneEntityCfg("/World/MyRobot")}
)

#End-effector(claw)
ee_pos: ObservationTermCfg = ObservationTermCfg(
    func="omni.isaac.lab.terms.ee_position",
    params={
        "asset_cfg": SceneEntityCfg("/World/MyRobot"),
        "ee_link_name": "GripperJoint"  # the name of your end-effector link
    }
)

cube_pos: ObservationTermCfg = ObservationTermCfg(
    func="omni.isaac.lab.terms.body_position",
    params={"rigid_body_cfg": SceneEntityCfg("/World/Cube")}
)

# Actions
class RobotPickActionCfg(ActionTermCfg):
    def __init__(self):
        super().__init__(
            func="omni.isaac.lab.actions.joint_position_delta",
            params={"asset_cfg": SceneEntityCfg("robot"), "scale": 0.05, "clip": 0.2},
        )
        #scale: Multiplies the RL action output to control step size (how big each movement is per timestep).
        #clip: Limits the maximum delta to prevent excessive joint movement in one step.


# ---------------- Rewards ----------------
class RobotPickRewardCfg:
    reaching = RewardTermCfg(
        func="omni.isaac.lab.rewards.distance",
        params={"src": "ee_pos", "tgt": "cube_pos"},
        weight=-1.0,
    )
    lift_bonus = RewardTermCfg(
        func="omni.isaac.lab.rewards.height_threshold",
        params={"entity": "cube_pos", "threshold": 0.20},
        weight=5.0,
    )


# ---------------- Terminations ----------------
class RobotPickTerminationCfg:
    time_out = TerminationTermCfg(
        func="omni.isaac.lab.terminations.time_out", params={"max_steps": 200}
    )
    success = TerminationTermCfg(
        func="omni.isaac.lab.terminations.height_threshold",
        params={"entity": "cube_pos", "threshold": 0.20},
    )


# ---------------- Env Config ----------------
class RobotPickEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotPickSceneCfg = RobotPickSceneCfg(num_envs=64, env_spacing=1.5)
    observations = {"policy": RobotPickObsCfg()}
    actions = {"arm_action": RobotPickActionCfg()}
    rewards = RobotPickRewardCfg()
    terminations = RobotPickTerminationCfg()


# ---------------- Env ----------------
class RobotPickEnv(ManagerBasedRLEnv):
    cfg: RobotPickEnvCfg

    def __init__(self, cfg: RobotPickEnvCfg, render: bool = False, **kwargs):
        super().__init__(cfg, render=render, **kwargs)
