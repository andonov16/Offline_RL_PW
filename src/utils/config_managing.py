import os
import sys
import json


def create_bc_config_file(bc_config_file_path: str = 'D:/University/5th_Semester/Practical_Work_in_AI/Offline_RL_PW/config/bc_config.json') -> None:
    config_dir = os.path.dirname(bc_config_file_path)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    bc_config_dict = {
        'seed': 16,
        'train_ratio': 0.70,
        'test_ratio': 0.15,
        'valid_ratio': 0.15,
        'batch_size': 32,
        'epochs': 100,
        'log_subfolder': 'logs/bc',
        'early_stop_epoch_without_improvement': 5,
        'input_neurons': 8,
        'hidden_neurons': [32, 64, 128],
        'num_hidden_layers': [5, 7, 9, 12, 15],
        'out_neurons': 4,
        'lr': [0.01, 0.001, 0.0001],
        'weight_decay': [0, 1e-5, 1e-4]
    }

    with open(bc_config_file_path, 'w') as f:
        json.dump(bc_config_dict, f, indent=4)


def load_bc_config_file(bc_config_file_path: str = 'D:/University/5th_Semester/Practical_Work_in_AI/Offline_RL_PW/config/bc_config.json') -> dict:
    if not os.path.exists(bc_config_file_path):
        create_bc_config_file(bc_config_file_path)

    with open(bc_config_file_path, 'r') as f:
        bc_config_dict = json.load(f)

    return bc_config_dict


if __name__ =='__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname("__file__"), "../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)
    bc_config_hyperparams = load_bc_config_file()
    print(bc_config_hyperparams)
