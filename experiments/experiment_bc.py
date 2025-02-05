from typing import Tuple
from tqdm import tqdm
import warnings
import torch

from src.behavior_cloning import BC
from src.utils.training import *
from src.utils.data_loading import *

device = torch.device('cuda')
if "cuda" in device.type and not torch.cuda.is_available():
    warnings.warn("CUDA not available, falling back to CPU")
    device = torch.device("cpu")

model = BC(8, 16, 5, 4)

train_loader, test_loader, valid_loader = get_BC_data_loaders()

training_loop(network=model,
              train_loader=train_loader,
              eval_loader=valid_loader,
              max_epochs=5,
              device=device)

