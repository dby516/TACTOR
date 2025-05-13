# TACTOR: Tactile-based Online Active Shape Exploration and Reconstruction

This repository contains the implementation of TACTOR, a reinforcement learning framework for online active shape exploration using tactile sensing. The system uses a single-finger tactile sensor and a PPO-based policy to explore and reconstruct object shapes.

## Features

- MuJoCo-based tactile sensor simulation
- PointNet-based point cloud processing
- PPO reinforcement learning for exploration policy
- Multi-modal observation processing (tactile, point cloud, pose)
- Configurable reward functions (contact, voxel occupancy, Chamfer distance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tactor.git
cd tactor
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the agent:

```bash
python src/train.py
```

The training configuration can be modified in `src/configs/training_config.yaml`.

### Environment

The environment is implemented in `src/environment/tactile_env.py` and uses a MuJoCo model defined in `src/environment/tactile_sensor.xml`. The environment provides:

- 6-DoF action space for robot control
- Multi-modal observations (tactile readings, point cloud, robot pose)
- Reward functions for contact, voxel occupancy, and Chamfer distance

### Models

- `src/models/pointnet.py`: PointNet implementation for point cloud processing
- `src/models/ppo_policy.py`: PPO policy network with multi-modal input processing

## Configuration

The system can be configured through `src/configs/training_config.yaml`:

- Environment parameters (sensor configuration, step limits)
- Training hyperparameters (learning rate, batch size, etc.)
- Model architecture (network dimensions, dropout rates)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{tactor2024,
  title={TACTOR: Tactile-based Online Active Shape Exploration and Reconstruction},
  author={Du, Bingyao and Zou, Hao and Zhang, Linlin},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 