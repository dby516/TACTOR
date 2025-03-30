import torch
import numpy as np
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.ur10e_tactip import UR10e_TACTIP_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


class TACTORScene:
    def __init__(self, shape_usd_list: list[str], spacing: float = 2.5):
        """
        Initialize the TACTOR scene with 6 workspaces.
        Each includes a table, a UR10e+TacTip robot, and a custom object.
        """
        assert len(shape_usd_list) == 6, "Provide exactly 6 shape USDs (one per origin)"

        self.shape_usd_list = shape_usd_list
        self.spacing = spacing
        self.scene_entities = {}  # Stores each robot articulation
        self.origins = self._define_origins(num_origins=6, spacing=spacing)

        self._setup_environment()

    def _define_origins(self, num_origins: int, spacing: float) -> list[list[float]]:
        """
        Compute grid-based spatial origins for each environment.
        Returns a list of [x, y, z] coordinates.
        """
        env_origins = torch.zeros(num_origins, 3)
        num_rows = int(np.floor(np.sqrt(num_origins)))
        num_cols = int(np.ceil(num_origins / num_rows))
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
        env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
        env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
        env_origins[:, 2] = 0.0  # All environments are on the ground plane
        return env_origins.tolist()

    def _setup_environment(self):
        """
        Populate the scene with lighting, ground, and 6 distinct robot-table-object environments.
        """
        # Add a ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # Add a dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Loop through each origin and place the environment components
        for i in range(6):
            origin = self.origins[i]
            ns = f"/World/Origin{i}"

            # Create a named transform group at the origin
            prim_utils.create_prim(ns, "Xform", translation=origin)

            # Define local positions for each component, offset from origin
            table_pos = (origin[0], origin[1], origin[2] + 0.8)        # Table sits on the ground
            robot_pos = (origin[0], origin[1], origin[2] + 0.8)        # Robot sits on the table
            object_pos = (origin[0], origin[1] + 0.25, origin[2] + 0.85)  # Object slightly in front of robot

            # Spawn the table
            table_cfg = sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
            )
            table_cfg.func(f"{ns}/Table", table_cfg, translation=table_pos)

            # Spawn the robot
            robot_cfg = UR10e_TACTIP_CFG.replace(prim_path=f"{ns}/Robot")
            robot_cfg.init_state.pos = robot_pos
            robot = Articulation(cfg=robot_cfg)
            self.scene_entities[f"ur10e_{i}"] = robot  # Store reference for external access

            # Spawn the object (unique USD per origin)
            object_cfg = sim_utils.UsdFileCfg(
                usd_path=self.shape_usd_list[i],
                scale=(50.0, 50.0, 50.0)
            )
            object_cfg.func(f"{ns}/Object", object_cfg, translation=object_pos)

    def get_scene_entities(self):
        """Returns the dictionary of robot articulations."""
        return self.scene_entities

    def get_origins(self):
        """Returns the list of origin coordinates."""
        return self.origins
