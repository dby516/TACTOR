"""
Usage:
./external_tools/IsaacLab/isaaclab.sh -p scripts/train.py --task Isaac-Tactor-RL-v0 --num_envs 64 --headless --video
"""
import argparse
from utils import cli_args
from isaaclab.app import AppLauncher

import os
import sys
import torch
from datetime import datetime



# ----------------- Argument parsing ------------------
parser = argparse.ArgumentParser(description="Train the TACTOR policy for tactile shape exploration.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# -------------- Launch simulator ------------------
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable fast math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


import gymnasium as gym

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.utils.dict import print_dict

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import tactor_rl.tasks.tactor, tactor_rl.tasks.shape_explore
import tactor_rl.networks

# ----------------- Main Training Logic ------------------
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg | ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations or agent_cfg.max_iterations
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[TACTOR] Logging to: {log_dir}")

    # Make gym-compatible task from IsaacLab
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Video recorder
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[TACTOR] Video recording enabled.")
        print_dict(video_kwargs)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    print("[INFO] Finish Wrapping for RSL-RL")

    # ------------ Use TACTOR policy ------------
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[TACTOR] Resuming from checkpoint: {resume_path}")
        runner.load(resume_path)

    # Save configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Train!
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
