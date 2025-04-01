import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Tactor-RL-v0",
    entry_point="tactor_rl.tasks.tactor_rl_task:TactorRLTask",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_tasks.direct.tactor_rl.tactor_rl_env_cfg:TactorRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"tactor_rl.tasks.agents.rsl_rl_ppo_cfg:TactorPPORunnerCfg",
    },
)
