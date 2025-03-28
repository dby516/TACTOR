"""
Usage: ./utils/IsaacLab/isaaclab.sh -p scripts_lab/build_sensor.py /home/bingyao/tactileX/assets_usds/standard_tactip.usd /home/bingyao/tactileX/assets_usds/standard_tactip_with_sensors.usd --num_sensors 32

"""
import argparse
import contextlib
import os
import math
import torch

from isaaclab.app import AppLauncher

# CLI Args
parser = argparse.ArgumentParser("Adds contact sensors over a hemisphere on a USD asset.")
parser.add_argument("input_usd", type=str, help="Path to original USD file.")
parser.add_argument("output_usd", type=str, help="Path to save the modified USD file.")
parser.add_argument("--radius", type=float, default=0.020, help="Sensor hemisphere radius.")
parser.add_argument("--offset", type=float, default=0.065, help="Distance from base to sphere center.")
parser.add_argument("--num_sensors", type=int, default=64, help="Number of sensors to place.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.usd
import omni.kit.app
import carb
from pxr import Usd, UsdGeom, UsdPhysics, Gf, PhysxSchema
import isaacsim.core.utils.stage as stage_utils


def generate_hemisphere_sensor_poses(radius, center_offset, num_points):
    golden_ratio = (1 + 5 ** 0.5) / 2
    indices = torch.arange(0, num_points)
    theta = 2 * math.pi * indices / golden_ratio
    phi = torch.acos(1 - indices / num_points)
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi) + center_offset
    return torch.stack([x, y, z], dim=-1)


def main():
    usd_path = os.path.abspath(args.input_usd)
    stage_utils.open_stage(usd_path)
    stage = omni.usd.get_context().get_stage()

    # Create articulation root
    root_path = "/Root"
    root_prim = stage.DefinePrim(root_path, "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    # Create base link
    base_path = f"{root_path}/standard_tactip"
    base_prim = stage.DefinePrim(base_path, "Xform")
    # UsdPhysics.RigidBodyAPI.Apply(base_prim)

    # Generate sensor positions
    sensor_positions = generate_hemisphere_sensor_poses(args.radius, args.offset, args.num_sensors)

    for i, pos in enumerate(sensor_positions):
        sensor_path = f"{base_path}/sensor_{i}"
        geom_path = f"{sensor_path}/geom"

        # Create Xform
        xform = UsdGeom.Xform.Define(stage, sensor_path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))

        # Create tiny sphere
        sphere = UsdGeom.Sphere.Define(stage, geom_path)
        sphere.GetRadiusAttr().Set(0.001)

        # Enable physics
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(sensor_path))
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(geom_path))
        UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(geom_path))

        # Add fixed joint between base and sensor
        joint_path = f"{sensor_path}/joint"
        
        # Create fixed joint
        joint_prim = stage.DefinePrim(joint_path, "PhysicsJoint")
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([base_path])
        joint.CreateBody1Rel().SetTargets([sensor_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Save to new file
    out_path = os.path.abspath(args.output_usd)
    stage.Export(out_path)
    print(f"[INFO]: Saved modified USD to {out_path}")

    # Optional GUI view
    carb_settings_iface = carb.settings.get_settings()
    if carb_settings_iface.get("/app/window/enabled") or carb_settings_iface.get("/app/livestream/enabled"):
        app = omni.kit.app.get_app_interface()
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
