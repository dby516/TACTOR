import torch
import numpy as np
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs.scene import BaseScene
from isaaclab.sim.spawners import UsdFileCfg
from tactor_rl.envs.tactor_env_cfg import TactorEnvCfg


class TacShapeExploreEnv(BaseScene):
    def __init__(self, cfg: TactorEnvCfg, spacing: float = 2.5):
        super().__init__()
        self.cfg = cfg
        self.spacing = spacing
        self.origins = self._define_origins(num_origins=6, spacing=spacing)
        self._setup_scene()

    def _define_origins(self, num_origins: int, spacing: float) -> list[list[float]]:
        env_origins = torch.zeros(num_origins, 3)
        num_rows = int(np.floor(np.sqrt(num_origins)))
        num_cols = int(np.ceil(num_origins / num_rows))
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
        env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
        env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
        env_origins[:, 2] = 0.0
        return env_origins.tolist()

    def _setup_scene(self):
        # Ground
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Environments
        for i, origin in enumerate(self.origins):
            ns = f"/World/envs/env_{i}"
            prim_utils.create_prim(ns, "Xform", translation=origin)

            table_pos = (origin[0], origin[1], origin[2] + 0.8)
            robot_pos = (origin[0], origin[1], origin[2] + 0.8)
            object_pos = (origin[0], origin[1] + 0.25, origin[2] + 0.85)

            # Table
            table_cfg = self.cfg.table_cfg.spawn
            table_cfg.func(f"{ns}/Table", table_cfg, translation=table_pos)

            # Robot
            robot_cfg = self.cfg.robot_cfg.replace(prim_path=f"{ns}/Robot")
            robot_cfg.init_state.pos = robot_pos
            robot = Articulation(cfg=robot_cfg)
            self.articulations[f"ur10e_{i}"] = robot

            # Object
            object_usd = self.cfg.object_usd_list[i]
            object_cfg = UsdFileCfg(
                usd_path=object_usd,
                scale=self.cfg.object_scale
            )
            object_cfg.func(f"{ns}/Object", object_cfg, translation=object_pos)

    def get_scene_entities(self):
        return self.articulations

    def get_origins(self):
        return self.origins
