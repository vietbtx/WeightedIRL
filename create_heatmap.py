from gail_airl_ppo.env import make_env
import numpy as np
import matplotlib.pyplot as plt
from airl_envs.twod_mjc_env import MapConfig, make_heat_map

def plot_heatmap(value_func, out_name, maze_left=False, show_title=False, num=128):
    config = MapConfig((-0.055, 0.55), (-0.055, 0.55), num, num)
    harvest = make_heat_map(value_func, config)
    plt.imshow(harvest)

    plt.scatter(0.5*num, 0.1*num-1, marker="*", s=250, c="lime", edgecolors='k')
    plt.scatter(0.5*num, 0.8*num-1, marker=".", s=500, c="w", edgecolors='k')

    if maze_left:
        plt.plot((0, 0.6*num), (0.5*num, 0.5*num), 'k', marker='o', linewidth=14)
    else:
        plt.plot((num-1, 0.4*num), (0.5*num, 0.5*num), 'k', marker='o', linewidth=14)

    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    if show_title:
        parts = out_name.replace("\\", "/").split("/")
        plt.title(f"{parts[-3]} - {parts[-1]}")
    plt.savefig(f"{out_name}.jpg", bbox_inches='tight')
    plt.savefig(f"{out_name}.pdf", bbox_inches='tight')
    plt.close()

