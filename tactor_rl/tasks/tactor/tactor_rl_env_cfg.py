from isaaclab_assets.robots.ur10e_tactip import UR10e_TACTIP_CFG, TACTIP_CFG
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg, UsdFileCfg, CuboidCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class TactorEnvSceneCfg(InteractiveSceneCfg):
    # Robot
    robot = TACTIP_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Object: replace it with ShapeNet object
    object = AssetBaseCfg(
        prim_path = "/World/envs/env_.*/Object",
        spawn = sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 1.0, 0.0)),
        ),
        init_state = AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
    )
    
    # Sensors

@configclass
class TactorEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    action_space = 7  # 3 translation + 4 rotation
    observation_space = 327  # for actor: 327, for critic: 576
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/1200,
        render_interval=decimation,
        physx=PhysxCfg(
            # min_position_iteration_count=2,
            # max_position_iteration_count=2,
            # min_velocity_iteration_count=0,
            # max_velocity_iteration_count=0,
            enable_ccd=False,
            enable_stabilization=True,
            enable_enhanced_determinism=False,
        )
    )

    # scene
    for i in range(64): # Add contact sensors
        setattr(TactorEnvSceneCfg, f"contact_sensor_{i}", ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/sensor_{i}",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            track_pose=True,
        ))
    scene: TactorEnvSceneCfg = TactorEnvSceneCfg(num_envs=6, env_spacing=2.5, replicate_physics=True)

    
    # reset (no joint reset needed for TacTip)
    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.0
    reset_dof_vel_noise = 0.0
