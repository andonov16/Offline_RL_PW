import numpy as np

def load_data(data_path: str = '../data/replay_buffer.npz') -> tuple:
    """
    Loads data from a saved replay buffer file in NumPy .npz format.

    Returns:
        - `observations`: np.array storing space observation vectors at each timestep. Shape: (1000000, 8), dtype: float32.
        - `next_observations`: np.array storing the next state reached after taking 
        the given action at each timestep. Shape: (1000000, 8), dtype: float32.
        - `actions`: np.array storing the action taken at each timestep. Shape: (1000000, 1), dtype: int64.
        - `rewards`: np.array storing the reward received at each timestep. Shape: (1000000, 1), dtype: float32.
        - `dones`: np.array indicating whether the episode terminated at each timestep
        either due to crashing the lander or landing successfully. Shape: (1000000, 1), dtype: float32.
    """
     
    with np.load(data_path) as data:
        observations = data['arr_0']
        next_observations = data['arr_1']
        actions = data['arr_2']
        rewards = data['arr_3']
        dones = data['arr_4']

        # remove the 'empty' dimentions
        observations = np.squeeze(observations, axis=1)
        next_observations = np.squeeze(next_observations, axis=1)
        actions = np.squeeze(actions, axis=2)
        
    return (observations, next_observations, actions, rewards, dones) 