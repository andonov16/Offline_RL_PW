import numpy as np
import os
from typing import Tuple

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.datasets import BCDataset


def load_data(data_path: str = '../data/replay_buffer.npz') -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Loads data from a saved replay buffer file in NumPy .npz format.

    Accepts:
        - 'data_path': str the path to the data stored in a file format .npz or .npy

    Returns:
        - `observations`: np.array storing space observation vectors at each timestep. Shape: (1000000, 8), dtype: float32.
        - `next_observations`: np.array storing the next state reached after taking 
        the given action at each timestep. Shape: (1000000, 8), dtype: float32.
        - `actions`: np.array storing the action taken at each timestep. Shape: (1000000, 1), dtype: int64.
        - `rewards`: np.array storing the reward received at each timestep. Shape: (1000000, 1), dtype: float32.
        - `dones`: np.array indicating whether the episode terminated at each timestep
        either due to crashing the lander or landing successfully. Shape: (1000000, 1), dtype: float32.
    """
    assert os.path.exists(data_path), 'The path to the dataset does not exist! Path:' + data_path
    assert os.path.isfile(data_path), 'The specified path is not a file! Path:' + data_path
    assert os.path.splitext(data_path)[1] in {'.npy', '.npz'}, 'Invalid file extension (must be either .npy or .npz! Path:' + data_path
     
    with np.load(data_path) as data:
        observations = data['arr_0']
        next_observations = data['arr_1']
        actions = data['arr_2']
        rewards = data['arr_3']
        dones = data['arr_4']

        # remove the 'empty' dimensions
        observations = np.squeeze(observations, axis=1)
        next_observations = np.squeeze(next_observations, axis=1)
        actions = np.squeeze(actions, axis=2)
        
    return observations, next_observations, actions, rewards, dones


def get_BC_data_loaders(train: float = 0.70,
                        test: float = 0.15,
                        validation: float = 0.15,
                        batch_size: int = 32,
                        seed: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
    assert abs(train + test + validation - 1.0) < 1e-5, 'Data splits must add up to 1.'

    observations, _, actions, _, _ = load_data()
    np.random.seed(seed)
    test_to_valid_ratio = test / (test + validation)

    observations_train, observations_test_valid, actions_train, actions_test_valid = train_test_split(
        observations, actions, test_size=(1 - train), random_state=seed)
    observations_test, observations_valid, actions_test, actions_valid = train_test_split(
        observations, actions, test_size=test_to_valid_ratio, random_state=seed)

    train_dataset = BCDataset(observations_train, actions_train)
    test_dataset = BCDataset(observations_test, actions_test)
    valid_dataset = BCDataset(observations_valid, actions_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, valid_loader
