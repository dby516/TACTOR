"""
This script demonstrates a Tactip robot moving along a predefined trajectory over a sphere surface,
while visualizing contact sensor readings in 3D space.

Usage:
    ./utils/IsaacLab/isaaclab.sh -p ./data_collection/test_contact.py --headless
    ffmpeg -framerate 10 -i frame3d_%04d.png -c:v libx264 -pix_fmt yuv420p contact_3d_forces.mp4
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from isaaclab.app import AppLauncher
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
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from copy import deepcopy
from contact_data import ContactProcessor

@configclass
class ContactHandSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = deepcopy(TACTIP_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    sphere = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SphereGround",
        spawn=sim_utils.MeshSphereCfg(
            radius=1.0,
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.05, 1.0)),
    )

def generate_sensor_names(num_sensors=64):
    return [f"contact_sensor_{i}" for i in range(num_sensors)]

os.makedirs("/home/bingyao/tactileX/outputs/frames_3d", exist_ok=True)

def save_3d_quiver_plot(processor, frame_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for past_idx, (positions, forces) in enumerate(processor.history):
        positions = positions[0].cpu().numpy()
        forces = forces[0].cpu().numpy()
        alpha = 0.2 if past_idx < len(processor.history) - 1 else 1.0
        ax.quiver(
            positions[:, 0], positions[:, 1], positions[:, 2],
            forces[:, 0], forces[:, 1], forces[:, 2],
            length=0.01, normalize=True, color='red', alpha=alpha
        )
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0.0, 2.2)
    ax.set_title(f"Frame {frame_idx} (Force magnitude range: 0–230)")
    plt.savefig(f"/home/bingyao/tactileX/outputs/frames_3d/frame3d_{frame_idx:04d}.png", dpi=120)
    plt.close()


def orientation_toward_center(pos):
    forward = -F.normalize(pos, dim=1)  # point toward origin
    up = torch.tensor([[0.0, 0.0, 1.0]], device=pos.device).expand_as(forward)
    right = F.normalize(torch.cross(up, forward, dim=1), dim=1)
    up = torch.cross(forward, right, dim=1)

    # Construct rotation matrix [right; up; forward]
    rot_mat = torch.stack([right, up, forward], dim=-1)  # [B, 3, 3]
    return torch.tensor([
        [torch.cos(theta/2), axis[0]*torch.sin(theta/2), axis[1]*torch.sin(theta/2), axis[2]*torch.sin(theta/2)]
        for axis, theta in [rotation_matrix_to_axis_angle(R) for R in rot_mat]
    ], device=pos.device)

def rotation_matrix_to_axis_angle(R):
    theta = torch.acos((R.trace() - 1) / 2)
    axis = 1 / (2 * torch.sin(theta)) * torch.tensor([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ], device=R.device)
    return axis, theta

def random_sphere_trajectory(step):
    theta = (step % 300) / 300 * 2 * math.pi
    phi = math.pi / 3 + 0.1 * math.sin(step / 20.0)
    r = 1.065
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    pos = torch.tensor([[x, y, z]], device="cuda")
    ori = orientation_toward_center(pos)
    pos = torch.tensor([[x, y, z+1]], device="cuda")
    return pos, ori

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    count = 0
    scene["robot"]._has_implicit_actuators = False
    sensor_names = generate_sensor_names()
    processor = ContactProcessor(sensor_names=sensor_names, scene=scene, device=sim.device)

    while simulation_app.is_running():
        # if count % 100 == 0:
        #     scene.reset()
        #     print("[INFO]: Scene reset at step", count)
        pos, ori = random_sphere_trajectory(count)
        root_state = scene["robot"].data.default_root_state.clone()
        root_state[:, :3] = pos + scene.env_origins
        root_state[:, 3:7] = ori
        scene["robot"].write_root_pose_to_sim(root_state[:, :7])
        scene["robot"].write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))

        scene["robot"].write_data_to_sim()
        scene["sphere"].write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        magnitudes = processor.get_force_magnitudes()
        processor.get_force_vectors_in_space()
        max_force = magnitudes.max().item()
        print(f"Step {count:03d} - Max Contact Force: {max_force:.4f}")
        save_3d_quiver_plot(processor, count)

        count += 1
        if count >= 200:
            print("[INFO]: Reached 200 steps. Shutting down...")
            break

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
            filter_prim_paths_expr=["{ENV_REGEX_NS}/SphereGround"],
        ))
    scene_cfg = ContactHandSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)
    print("[INFO]: Simulation complete.")
    simulation_app.close()

if __name__ == "__main__":
    main()
