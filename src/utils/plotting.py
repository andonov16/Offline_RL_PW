import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_hist(data_dict: dict, num_columns: int = 2, f_size: tuple = (16, 9), 
        custom_title: str = None, single_ax: bool = False, hist_kwargs: dict = None) -> plt.Figure:
    if single_ax:
        return _plot_hist_single_axis(data_dict=data_dict, f_size=f_size, custom_title=custom_title, hist_kwargs=hist_kwargs)
    return _plot_hist_multiple_axes(data_dict=data_dict, num_columns=num_columns, f_size=f_size, custom_title=custom_title, hist_kwargs=hist_kwargs)


def _plot_hist_single_axis(data_dict: dict, f_size: tuple = (16, 9), custom_title: str = None, hist_kwargs: dict = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=f_size)

    if custom_title is None:
        ax.set_title('Histogram')
    else:
        ax.set_title(custom_title)

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Data')

    for label, data in data_dict.items():
        ax.hist(data, label=label, **hist_kwargs)

    ax.legend()
    fig.tight_layout()

    return fig
    

def _plot_hist_multiple_axes(data_dict: dict, num_columns: int = 2, f_size: tuple = (16, 9), custom_title: str = None, hist_kwargs: dict = None) -> plt.Figure:
    num_rows = int(np.ceil(len(data_dict.keys())/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    for ax, data_name in zip(axes, data_dict.keys()):
        if custom_title is None:
            ax.set_title('Histogram: '+ data_name)
        else:
            ax.set_title(custom_title)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Data')
        ax.hist(data_dict[data_name], **hist_kwargs)
    
    fig.tight_layout()
    
    return fig



def plot_hexbin(data_dict: dict, num_columns: int = 2, f_size: tuple = (16,9)) -> plt.Figure:
    num_rows = int(np.ceil(len(data_dict.keys())/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    all_x = np.concatenate([data[:, 0] for data in data_dict.values()])
    all_y = np.concatenate([data[:, 1] for data in data_dict.values()])
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    for ax, data_name in zip(axes, data_dict.keys()):
        ax.set_title('Hexbin map: '+ data_name)
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        hb = ax.hexbin(x=data_dict[data_name][:, 0], y=data_dict[data_name][:, 1], cmap='viridis', mincnt=1)
    
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        fig.colorbar(hb, ax=ax, orientation='vertical')
    
    fig.tight_layout()
    
    return fig
    
    
def plot_kde_density(data_dict: dict, num_columns: int = 2, f_size: tuple = (16, 9)) -> plt.Figure:
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


def plot_boxplots_df(data: pd.DataFrame, num_columns: int = 4, f_size: tuple = (16, 9), custom_title: str = None, boxplot_kwargs: dict = None) -> plt.Figure:
    num_rows = int(np.ceil(len(data.columns)/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    if custom_title is None:
        fig.suptitle(f'Boxplots')
    else:
        fig.suptitle(custom_title)
    
    for i, column in enumerate(data.columns):
        if boxplot_kwargs is None:
            data.boxplot(column=[column], ax=axes[i])
        else:
            data.boxplot(column=[column], ax=axes[i], **boxplot_kwargs)
        axes[i].set_title(f'Boxplot of {column}')
    fig.tight_layout()
    return fig


def plot_boxplots_compare_df(df1_name: str, df2_name: str, df1: pd.DataFrame, df2: pd.DataFrame, num_columns: int = 4,
                            f_size: tuple = (16, 9), custom_title: str = None, boxplot_kwargs: dict = None) -> plt.Figure:
    num_rows = int(np.ceil(len(df1.columns)/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    df1_renamed = df1.rename(columns={col: f"{df1_name}_{col}" for col in df1.columns})
    df2_renamed = df2.rename(columns={col: f"{df2_name}_{col}" for col in df2.columns})
    df_merged = pd.concat([df1_renamed, df2_renamed], axis=1)
    
    if custom_title is None:
        fig.suptitle(f'Boxplots')
    else:
        fig.suptitle(custom_title)
    
    for i, column in enumerate(df1.columns):
        if boxplot_kwargs is None:
            df_merged.boxplot(column=[df1_name+'_'+column, df2_name+'_'+column], ax=axes[i])
        else:
            df_merged.boxplot(column=[df1_name+'_'+column, df2_name+'_'+column], **boxplot_kwargs)
        axes[i].set_title(f'Boxplot of {column}')
    fig.tight_layout()
    return fig
