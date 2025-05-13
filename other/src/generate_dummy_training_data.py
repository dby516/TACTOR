import numpy as np
import os

def main():
    os.makedirs('logs', exist_ok=True)
    n_episodes = 200
    n_steps = 300
    n_sensors = 16
    n_actions = 6
    rewards = np.cumsum(np.random.randn(n_episodes)) + 50
    contacts = np.clip(np.random.rand(n_episodes), 0, 1)
    coverage = np.clip(np.cumsum(np.random.rand(n_episodes) * 0.01), 0, 1)
    actions = np.random.randn(n_episodes, n_actions)
    point_clouds = np.random.randn(n_episodes, 100, 3)
    tactile = np.abs(np.random.randn(n_steps, n_sensors))
    np.savez('logs/training_data_real.npz',
             rewards=rewards,
             contacts=contacts,
             coverage=coverage,
             actions=actions,
             point_clouds=point_clouds,
             tactile=tactile)
    print('Dummy training_data_real.npz created.')

if __name__ == '__main__':
    main() 