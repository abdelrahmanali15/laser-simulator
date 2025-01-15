import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np


def setup_plotting_style():
    """
    Set up consistent plotting style with custom color scheme.
    Returns the color list for other uses.
    """
    colors = [
        '#4E79A7',  # Muted blue
        '#76B7B2',  # Teal
        '#59A14F',  # Green
        '#E15759',  # Coral red
        '#AF7AA1',  # Purple
        '#FF9DA7',  # Pink
        '#9C755F',  # Brown
        '#BAB0AC'   # Gray
    ]

    # Set up the color cycler globally
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)

    # Define common figure sizes based on golden ratio
    width = 7.2
    height = width / 1.618

    # Set other common plot parameters
    mpl.rcParams['figure.figsize'] = (width, height)
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.titlesize'] = 12

    return colors


def create_figure(rows=1, cols=1, is_subplot=False):
    """
    Create a figure with consistent sizing and offset color cycling for subplots.

    Parameters:
    rows (int): Number of subplot rows
    cols (int): Number of subplot columns
    is_subplot (bool): If True, uses larger height for subplots

    Returns:
    fig, ax: Figure and axes objects
    """
    width = 7.2
    height = width / 1.618

    if is_subplot:
        height *= rows  # Increase height for subplots

    fig, ax = plt.subplots(rows, cols, figsize=(width, height))

    # Get the color cycle
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    num_colors = len(colors)

    # Handle both single subplot and multiple subplot cases
    if isinstance(ax, np.ndarray):
        # For multiple subplots
        for idx, ax_item in enumerate(ax.flat):
            # Rotate colors by idx positions for each subplot
            rotated_colors = colors[idx:] + colors[:idx]
            ax_item.set_prop_cycle(cycler(color=rotated_colors))
    else:
        # For single subplot
        ax.set_prop_cycle(cycler(color=colors))

    return fig, ax


def style_axis(ax, title=None, xlabel=None, ylabel=None, legend=True):
    """
    Apply consistent styling to an axis.

    Parameters:
    ax: Matplotlib axis object or array of axes
    title (str): Plot title
    xlabel (str): X-axis label
    ylabel (str): Y-axis label
    legend (bool): Whether to show legend
    """
    # Handle both single axis and multiple axes cases
    if isinstance(ax, np.ndarray):
        axes = ax.flat
    else:
        axes = [ax]

    for ax_item in axes:
        ax_item.grid(True, which="both", ls="-", alpha=0.2)
        ax_item.grid(True, which="major", ls="-", alpha=0.5)

        if title:
            ax_item.set_title(title)
        if xlabel:
            ax_item.set_xlabel(xlabel)
        if ylabel:
            ax_item.set_ylabel(ylabel)
        if legend:
            ax_item.legend(fontsize=10, loc='best')
