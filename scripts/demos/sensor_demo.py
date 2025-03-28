"""
This script demonstrates different dexterous hands and records contact sensor plots.

Usage:
    ./isaaclab.sh -p scripts/demos/hands_contact.py
    ffmpeg -framerate 10 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p contact_forces.mp4
"""

import argparse
import os
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from matplotlib import cm

from isaaclab.app import AppLauncher
# --- CLI Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.ur10e_tactip import TACTIP_CFG
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from copy import deepcopy

# --- Config ---
@configclass
class ContactHandSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = deepcopy(TACTIP_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/bingyao/tactileX/objects/one.usd",
            scale=[12.0, 12.0, 4.0],
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_linear_velocity=10.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/standard_tactip/tactip_body",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )

# --- Sensor Layout ---
def generate_hemisphere_sensor_poses(radius=0.02, center_offset=0.065, num_points=64):
    golden_ratio = (1 + 5 ** 0.5) / 2
    indices = torch.arange(0, num_points)
    theta = 2 * math.pi * indices / golden_ratio
    phi = torch.acos(1 - indices / num_points)
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi) + center_offset
    return torch.stack([x, y, z], dim=-1)

sensor_positions = generate_hemisphere_sensor_poses()
xy_positions = sensor_positions[:, :2].cpu().numpy()
xy_min = xy_positions.min(axis=0)
xy_max = xy_positions.max(axis=0)
xy_norm = (xy_positions - xy_min) / (xy_max - xy_min)
fig, ax = plt.subplots()
sc = ax.scatter(xy_norm[:, 0], xy_norm[:, 1], c=np.zeros(64), cmap=cm.Reds, s=200, vmin=0, vmax=20)
ax.set_aspect('equal')
ax.axis('off')
os.makedirs("/home/bingyao/tactileX/outputs/frames", exist_ok=True)

def update_plot(forces, frame_idx):
    sc.set_array(forces)
    plt.savefig(f"/home/bingyao/tactileX/outputs/frames/frame_{frame_idx:04d}.png", dpi=100)

# --- Simulator Loop ---
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    count = 0
    scene["robot"]._has_implicit_actuators = False
    # nodal_kinematic_target = scene["cube"].data.nodal_kinematic_target.clone()

    while simulation_app.is_running():
        if count % 100 == 0:
            root_state_robot = scene["robot"].data.default_root_state.clone()
            root_state_robot[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state_robot[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state_robot[:, 7:])
            root_state_cube = scene["cube"].data.default_root_state.clone()
            root_state_cube[:, :3] += scene.env_origins
            scene["cube"].write_root_pose_to_sim(root_state_cube[:, :7])
            scene["cube"].write_root_velocity_to_sim(root_state_cube[:, 7:])

            # # reset the nodal state of the object
            # nodal_state = scene["cube"].data.default_nodal_state_w.clone()
            # # apply random pose to the object
            # pos_w = torch.rand(scene["cube"].num_instances, 3, device=sim.device) * 0.1 + scene.env_origins
            # quat_w = math_utils.random_orientation(scene["cube"].num_instances, device=sim.device)
            # nodal_state[..., :3] = scene["cube"].transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

            # # write nodal state to simulation
            # scene["cube"].write_nodal_state_to_sim(nodal_state)

            # # Write the nodal state to the kinematic target and free all vertices
            # nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            # nodal_kinematic_target[..., 3] = 1.0
            # scene["cube"].write_nodal_kinematic_target_to_sim(nodal_kinematic_target)


            scene.reset()
            print("[INFO]: Resetting Tactip state...")

        scene["robot"].write_data_to_sim()
        scene["cube"].write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        forces = np.zeros(64)
        for i in range(64):
            f_all = scene[f"contact_sensor_{i}"].data.net_forces_w  # [num_envs, 3]
            forces[i] = torch.norm(f_all, dim=1).sum().item()

        update_plot(forces, count)

        count += 1
        if count >= 150:
            print("[INFO]: Reached 350 steps. Shutting down...")
            break

# --- Main ---
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    for i in range(64):
        setattr(ContactHandSceneCfg, f"contact_sensor_{i}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/standard_tactip/sensor_{i}",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
        ))
    scene_cfg = ContactHandSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)
    print("[INFO]: Simulation complete. Frame images saved.")
    plt.close()
    simulation_app.close()

if __name__ == "__main__":
    main()

