import matplotlib.pyplot as plt


def plot_config() -> None:
    """
    General styling for plots
    """    
    plt.grid(True)


def plot_colors(i: int = 0) -> str:
    """
    List of colors used for lines in plots
    """
    colors = ["plum", "red", "green", "blue"]

    i = i % len(colors)

    return colors[i]
