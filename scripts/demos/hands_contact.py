# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different dexterous hands.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/hands_contact.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

##
# Pre-defined configs
##
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG  # isort:skip
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG  # isort:skip
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
import omni


@configclass
class ContactHandSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on Allegro Hand"""
    # Ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot
    allegro = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Rigid Object
    cube = RigidObjectCfg(
        prim_path = "{ENV_REGEX_NS}/Cube",
        spawn = sim_utils.UsdFileCfg(
            usd_path = "/home/bingyao/tactileX/objects/one.usd",
            scale = [10.0, 10.0, 10.0],
            mass_props = sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_linear_velocity=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.03, -0.1, 0.56)),
    )
    # cube = sim_utils.UsdFileCfg(
    #     usd_path = "/home/bingyao/tactileX/objects/one.usd",
    #     scale = [10.0, 10.0, 10.0],
    #     mass_props = sim_utils.MassPropertiesCfg(mass=5.0),
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    # )
    # usd_object = RigidObject(cfg=usd_object_cfg)
    # cube = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Cube",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.1, 0.1, 0.1),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity = True,
    #             max_angular_velocity = 0.0,
    #             max_linear_velocity = 0.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.03, -0.1, 0.56)),
    # )


    # Sensor
    contact_forces_thumb = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_link_3",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        # filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    contact_forces_ring = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ring_biotac_tip",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        # filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    contact_forces_middle = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/middle_biotac_tip",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        # filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )






def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Start with hand open
    grasp_mode = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 1000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset robots
            root_state = scene["allegro"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["allegro"].write_root_pose_to_sim(root_state[:, :7])
            scene["allegro"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["allegro"].data.default_joint_pos.clone(),
                scene["allegro"].data.default_joint_vel.clone(),
            )
            scene["allegro"].write_joint_state_to_sim(joint_pos, joint_vel)
            # reset the internal state
            scene.reset()

            print("[INFO]: Resetting robots state...")

            
        # toggle grasp mode
        if count % 100 == 0:
            grasp_mode = 1 - grasp_mode
        # apply default actions to the hands robots
        # generate joint positions
        joint_pos_target = scene["allegro"].data.soft_joint_pos_limits[..., grasp_mode]
        # apply action to the robot
        scene["allegro"].set_joint_position_target(joint_pos_target)
        # write data to sim
        scene["allegro"].write_data_to_sim()

        # print contact read
        print("-----------------------------")
        print(scene["contact_forces_thumb"])
        print("Received force matrix of: ", scene["contact_forces_thumb"].data.force_matrix_w)
        print("Received contact force of: ", scene["contact_forces_thumb"].data.net_forces_w)
        print("-----------------------------")
        print(scene["contact_forces_ring"])
        print("Received force matrix of: ", scene["contact_forces_ring"].data.force_matrix_w)
        print("Received contact force of: ", scene["contact_forces_ring"].data.net_forces_w)
        print("-----------------------------")
        print(scene["contact_forces_middle"])
        print("Received force matrix of: ", scene["contact_forces_middle"].data.force_matrix_w)
        print("Received contact force of: ", scene["contact_forces_middle"].data.net_forces_w)

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.0, -0.5, 1.5], target=[0.0, -0.2, 0.5])
    # design scene
    scene_cfg = ContactHandSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
