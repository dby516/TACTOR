import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from environment.tactile_env import TactileEnv
from models.ppo_policy import TactileFeatureExtractor
from callbacks.visualization_callback import VisualizationCallback
import yaml
from typing import Dict
import logging

def make_env(config: Dict, rank: int, seed: int = 0, render_mode=None):
    """
    Create a single environment.
    
    Args:
        config: Configuration dictionary
        rank: Environment rank
        seed: Random seed
        render_mode: Optional render mode for visualization
        
    Returns:
        env: Environment instance
    """
    def _init():
        env = TactileEnv(config['environment'], render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def train(config: Dict):
    """
    Train the PPO agent.
    
    Args:
        config: Configuration dictionary
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create vectorized environment
    n_envs = config['training']['n_envs']
    env = SubprocVecEnv([
        make_env(config, i) for i in range(n_envs)
    ])
    
    # Create evaluation environment with rendering
    eval_env = DummyVecEnv([make_env(config, 0, render_mode='human')])
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config['training']['save_dir'],
        log_path=config['training']['log_dir'],
        eval_freq=config['training']['eval_freq'],
        deterministic=True,
        render=True  # Enable rendering during evaluation
    )
    
    # Create visualization callback
    vis_callback = VisualizationCallback()
    
    # Combine callbacks
    callbacks = [eval_callback, vis_callback]
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=lambda _: config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        n_epochs=config['training']['n_epochs'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_range=config['training']['clip_range'],
        clip_range_vf=config['training']['clip_range_vf'],
        normalize_advantage=True,
        ent_coef=config['training']['ent_coef'],
        vf_coef=config['training']['vf_coef'],
        max_grad_norm=config['training']['max_grad_norm'],
        use_sde=config['training']['use_sde'],
        sde_sample_freq=config['training']['sde_sample_freq'],
        target_kl=config['training']['target_kl'],
        tensorboard_log=config['training']['tensorboard_log'],
        policy_kwargs=dict(
            features_extractor_class=TactileFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128)
        ),
        verbose=1
    )
    
    # Train the model
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Save the final model
        model.save(os.path.join(config['training']['save_dir'], 'final_model'))
        env.close()
        eval_env.close()

if __name__ == "__main__":
    # Load configuration
    with open('src/configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Start training
    train(config) 