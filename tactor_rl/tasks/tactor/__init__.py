import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Tactor-RL-v0",
    # entry_point="isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
    entry_point="tactor_rl.tasks.shape_explore.shape_explore_env:TacShapeExploreEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tactor_rl_env_cfg:TactorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TactorPPORunnerCfg",
    },
)

print("[INFO] Finish Gym registering!")
