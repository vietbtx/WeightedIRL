from tensorboard.backend.event_processing import event_accumulator
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tsmoothie.smoother import LowessSmoother

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size'   : 22
})

def get_returns(env="Pendulum-v0", algo="gail", seed=2212):
    # print("reading ...", env, algo, f"logs/{env}/{algo}/seed{seed}/summary/*")
    tensorboard_path = glob.glob(f"logs/{env}/{algo}/seed{seed}/summary/*")[0]
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    steps = np.array([item.step for item in ea.Scalars('return/test')])
    scores = np.array([item.value for item in ea.Scalars('return/test')])

    steps = steps[steps <= 1000000]
    scores = scores[:len(steps)]

    x = scores[-10:]
    print(f"Env: {env} - Algo: {algo} - Score: {x.mean():.2f}+/-{x.max()-x.mean():.2f}")
    smooths = LowessSmoother(0.2, 1).smooth(scores).smooth_data[0]

    ids = np.array_split(np.arange(len(steps)), min(100, len(steps)))
    uppers = np.array([max(smooths[0], scores[0])] + [max(scores[id].max(), smooths[id].max()) for id in ids] + [max(smooths[-1], scores[-1])])
    lowers = np.array([min(smooths[0], scores[0])] + [min(scores[id].min(), smooths[id].min()) for id in ids] + [min(smooths[-1], scores[-1])])
    steps2 = np.array([steps[0]] + [np.mean(steps[id]) for id in ids] + [steps[-1]])

    uppers = LowessSmoother(0.1, 1).smooth(uppers).smooth_data[0]
    lowers = LowessSmoother(0.1, 1).smooth(lowers).smooth_data[0]

    return steps, steps2, lowers, uppers, smooths

if __name__ == "__main__":
    envs = [
        # "CustomAnt-v0DisabledAnt-v0",
        # "PointMazeLeft-v0",

        # "HalfCheetah-v2",
        # "Hopper-v3",
        # "Humanoid-v2",
        # "InvertedPendulum-v2",
        "Reacher-v2",
        "HumanoidStandup-v2",
        "Walker2d-v2",
        # "Ant-v2",
        # "Pendulum-v0",
    ]
    # algos = ["GAIL", "WGAIL", "AIRL", "WAIRL", "AIRL_State_Only", "WAIRL_State_Only"]
    algos = ["GAIL", "WGAIL", "AIRL", "WAIRL"]
    algo_colors = ['#41e1b9', '#4169e1', '#e1b941', '#e14169', '#b9e141', '#6941e1']

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.3, hspace=0.4)

    
    # axs = [y for x in axs for y in x]
    # print("axs:", axs)
    for env, ax in zip(envs, axs):
        ax.set_title(env, {'fontweight': 'bold'}, pad=16)
        for algo, color in zip(algos, algo_colors):
            steps, steps2, lowers, uppers, smooths = get_returns(env, algo.lower(), 2212)
            ax.plot(steps, smooths, color=color, linewidth=2, label=algo)
            ax.fill_between(steps2, lowers, uppers, color=color, alpha=0.15, linewidth=0)

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc=(0.75, 0.25))
    # axs[-1].set_visible(False)
    axs[0].legend()

    # plt.show()
    plt.savefig("graph_mujoco.pdf",  bbox_inches="tight")

