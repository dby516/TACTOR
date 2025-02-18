# README

## Setup

#### Isaac Sim & Isaac Lab Installation

##### On remote server

Create conda environment.

```
conda create -n env_isaacsim python=3.10
conda activate env_isaacsim
# CUDA 12
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Pip install Isaac Sim. Refer to [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html#isaac-sim-app-install-pip).

```
pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com
```

Install extensions (optional).

```
pip install isaacsim[extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com
```

Test:

```
isaacsim
```

Test in python script.

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
## perform any Isaac Sim / Omniverse imports after instantiating the class
```

Now install IsaacLab. Refer to [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

Clone git repo:

```
git clone git@github.com:isaac-sim/IsaacLab.git
```

Install environmental dependencies:

```
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

Install only one specific framework(optional):

```
./isaaclab.sh --install rl_games
```

The valid options are `rl_games`, `rsl_rl`, `sb3`, `skrl`, `robomimic`, `none`.



Run Lab scripts:

```
cd IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

Run RL training (in headless mode):

```
./isaaclab.sh -p scripts/reinforcement_learning/rl_game/train.py --task=Isaac-Ant-v0 --headless
```



##### On local machine

Refer to [Work Station Installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).

Download latest IsaacSim package. For [Windows/Linux](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html).

Setup workstation.

```
# For Windows
# Download the file into your_directory\
mkdir your_directory\isaacsim
cd your_directory\isaacsim
tar -xvzf "isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.windows-x86_64.release.zip"
cd isaacsim
post_install.bat
isaac-sim.selector.bat
```

You can run Isaac Sim Application by selecting **Isaac Sim Full** and clicking START.



#### Streaming

Download **Isaac Sim WebRTC Streaming Client** from the [Latest Release](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release) section for your platform.

Refer to [Livestream Clients](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html).

Try running Isaac Sim Streaming Server-Client on your local machine.

Run the Server:

```
cd the_directory_you_install_isaacsim\
isaac-sim.streaming.bat
```

Run the **Isaac Sim WebRTC Streaming Client** app and enter 127.0.0.1 as your server address.



If succeed, run the streaming Server on remote machine:

(for example, run create_empty.py)

```
cd IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --livestream 2
```

Then, run the Client app locally and enter your server IP as your server address.

