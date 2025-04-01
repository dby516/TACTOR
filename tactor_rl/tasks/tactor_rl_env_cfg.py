# tactor_rl/envs/tactor_env_cfg.py

from isaaclab.envs import SceneEntityCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.ur10e_tactip import UR10e_TACTIP_CFG
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class TactorEnvCfg:
    robot_cfg: ArticulationCfg = UR10e_TACTIP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
        ),
    )
    object_usd_list: list[str] = [
        f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
    ] * 6  # Placeholder, should be replaced with real ShapeNet paths
    object_scale: tuple[float, float, float] = (50.0, 50.0, 50.0)
