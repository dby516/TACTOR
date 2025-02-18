# README

## Setup Guide

Welcome! This guide will walk you through the installation of Isaac Sim and Isaac Lab on both remote servers and local machines. Follow these steps carefully to ensure a smooth setup experience.

### Isaac Sim & Isaac Lab Installation

#### Installing on a Remote Server

##### 1. Create a Conda Environment
To begin, set up a Conda environment with Python 3.10:
```bash
conda create -n env_isaacsim python=3.10
conda activate env_isaacsim
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

##### 2. Install Isaac Sim
You can find detailed instructions in the [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html#isaac-sim-app-install-pip).
```bash
pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com
```

If you need extensions, install them with:
```bash
pip install isaacsim[extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com
```

##### 3. Verify Installation
To ensure everything is installed correctly, run:
```bash
isaacsim
```

You can also verify it using a simple Python script:
```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
```

##### 4. Install Isaac Lab
Refer to the [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) for more details.
```bash
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install  # or "./isaaclab.sh -i"
```

To install a specific framework (optional):
```bash
./isaaclab.sh --install rl_games
```
Valid options: `rl_games`, `rsl_rl`, `sb3`, `skrl`, `robomimic`, `none`.

##### 5. Running Isaac Lab Scripts
```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

##### 6. Running RL Training in Headless Mode
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_game/train.py --task=Isaac-Ant-v0 --headless
```

---

#### Installing on a Local Machine

For workstation installation, please follow the [official guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).

##### 1. Download and Extract Isaac Sim
To get started, download Isaac Sim from the [official page](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html).

For Windows/Linux:
```bash
mkdir your_directory/isaacsim
cd your_directory/isaacsim
tar -xvzf "isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.windows-x86_64.release.zip"
cd isaacsim
post_install.bat
isaac-sim.selector.bat
```

To launch Isaac Sim, select **Isaac Sim Full** and click **START**.

---

### Setting Up Streaming

##### 1. Install the Streaming Client
Download the **Isaac Sim WebRTC Streaming Client** from the [latest release](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release).

For detailed guidance, refer to the [Livestream Clients Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html).

##### 2. Running the Streaming Server Locally
Run the following command to start the streaming server:
```bash
cd the_directory_you_install_isaacsim/
isaac-sim.streaming.bat
```

Launch the **Isaac Sim WebRTC Streaming Client**, and enter `127.0.0.1` as the server address.

##### 3. Running the Streaming Server on a Remote Machine
To run an example (e.g., `create_empty.py`):
```bash
cd IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --livestream 2
```

Now, launch the Client app locally and enter your server's IP address to connect.

---

If you encounter any issues, please refer to the official documentation or seek assistance from the community. Enjoy your experience with Isaac Sim and Isaac Lab!

