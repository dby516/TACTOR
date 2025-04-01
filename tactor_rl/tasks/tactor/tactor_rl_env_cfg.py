from isaaclab_assets.robots.ur10e_tactip import UR10e_TACTIP_CFG, TACTIP_CFG
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


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
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    # robot
    # robot_cfg: ArticulationCfg = UR10e_TACTIP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg: ArticulationCfg = TACTIP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot_cfg: ArticulationCfg = ALLEGRO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # contact sensors
    contact_sensors: list[ContactSensorCfg] = [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/standard_tactip/sensor_{i}",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
        )
        for i in range(64)
    ]

    # object placeholder config (will be overridden per-shape in TacShapeExploreEnv)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(5.0, 5.0, 5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_linear_velocity=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # table
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=6, env_spacing=2.5, replicate_physics=True)

    # reset (no joint reset needed for TacTip)
    reset_position_noise = 0.01
    reset_dof_pos_noise = 0.0
    reset_dof_vel_noise = 0.0
