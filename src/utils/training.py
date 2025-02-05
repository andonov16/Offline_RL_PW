from typing import Tuple
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from src.behavior_cloning import BC


def training_loop(
        network: BC,
        train_loader: torch.utils.data.DataLoader,
        eval_loader: torch.utils.data.DataLoader,
        max_epochs: int,
        device: torch.device,
        loss_func: callable = torch.nn.MSELoss(),
        show_progress: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5) -> Tuple[list, list, torch.nn.Module, float]:
    train_losses = list()
    eval_losses = list()
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    enumerator = range(max_epochs)
    if show_progress:
        enumerator = tqdm(enumerator, desc='Training ...')
    for epoch_num in enumerator:
        curr_train_losses = list()
        curr_eval_losses = list()

        # set the network to training mode
        network.train()
        for observations, actions in train_loader:
            observations = observations.float().to(device)
            actions = actions.to(device)

            output = network(observations)
            optimizer.zero_grad()
            loss = loss_func(output, actions)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            curr_train_losses.append(loss)

        network.eval()
        for observations, actions in eval_loader:
            observations = observations.float().to(device)
            actions = actions.to(device)

            output = network(observations)
            loss = loss_func(output, actions)
            curr_eval_losses.append(loss)

        if epoch_num > 3 and time_for_early_stopping(eval_losses):
            break

    eval_accuracy = calculate_accuracy(network, eval_loader)
    return train_losses, eval_losses, network, eval_accuracy


def calculate_accuracy(model: torch.nn.Module, eval_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for observations, actions in eval_loader:
            observations = observations.float().to(device)
            actions = actions.to(device)

            outputs = model(observations)
            _, predicted = torch.max(outputs, 1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()

    accuracy = correct / total * 100
    return accuracy


def time_for_early_stopping(eval_losses: list) -> bool:
    for i in range(1, 4):
        if eval_losses[-i].item() <= min([l.item() for l in eval_losses[::-3]]):
            return False
    return True
