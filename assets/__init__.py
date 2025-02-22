# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations."""

import os
import toml

"""Paths to Isaac Sim Assets"""
from isaaclab_asset import (
    ISAACLAB_ASSETS_DATA_DIR,
    ISAACLAB_ASSETS_EXT_DIR,
    ISAACLAB_ASSETS_METADATA,
    __version__
)

from isaaclab.utils.assets import (
    NUCLEUS_ASSET_ROOT_DIR,
    NVIDIA_NUCLEUS_DIR,
    ISAAC_NUCLEUS_DIR,
    ISAACLAB_NUCLEUS_DIR,
)

SAMPLE_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/Samples/"
"""Path to the ``Samples`` directory on the NVIDIA Nucleus Server."""

"""Customized Assets"""
from .robots import *
