import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_hist(data_dict: dict, num_columns: int = 2, f_size: tuple = (16, 9)) -> plt.figure:
    num_rows = int(np.ceil(len(data_dict.keys())/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    for ax, data_name in zip(axes, data_dict.keys()):
        ax.set_title('Histogram: '+ data_name)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Data')
        ax.hist(data_dict[data_name])
    
    fig.tight_layout()
    
    return fig


def plot_hexbin(data_dict: dict, num_columns: int = 2, f_size: tuple = (16,9)) -> plt.figure:
    num_rows = int(np.ceil(len(data_dict.keys())/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    all_x = np.concatenate([data[:, 0] for data in data_dict.values()])
    all_y = np.concatenate([data[:, 1] for data in data_dict.values()])
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    for ax, data_name in zip(axes, data_dict.keys()):
        ax.set_title('Hexboin map: '+ data_name)
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        hb = ax.hexbin(x=data_dict[data_name][:, 0], y=data_dict[data_name][:, 1], cmap='viridis', mincnt=1)
    
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        fig.colorbar(hb, ax=ax, orientation='vertical')
    
    fig.tight_layout()
    
    return fig
    
    
def plot_kde_density(data_dict: dict, num_columns: int = 2, f_size: tuple = (16, 9)) -> plt.figure:
    num_rows = int(np.ceil(len(data_dict.keys()) / num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    for ax, (data_name, data) in tqdm(zip(axes, data_dict.items())):
        ax.set_title(f'KDE plot: {data_name}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        sns.kdeplot(data, ax=ax, fill=True, thresh=0.1)
    
    fig.tight_layout()
    
    return fig
