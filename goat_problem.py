#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['font.size']=23
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
from pathlib import Path
path = Path('./total_probability/')
# %%
class UpdateFigure:
    def __init__(self, ax:plt.Axes, data:np.ndarray):
        self.data = data
        self.trials = np.arange(data.shape[0])+1
        self.rate = np.cumsum(data)/self.trials
        self.line, = ax.plot([], [], '-o', color='navy', markeredgecolor='navy', markerfacecolor='orange')
        self.line.set_clip_on(False)

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i > 0 and i <= self.data.shape[0]:
            self.line.set_data(self.trials[:i], self.rate[:i])
        return [self.line,]

def create_fig():
    fig, ax = plt.subplots(1,1,figsize=(8,3),
                        gridspec_kw={'left':0.15, 'bottom':0.23, 'right':0.95, 'top':0.95})
    ax.set_xlim(0,N)
    ax.set_ylim(0,1)
    ax.set_xlabel('游戏次数')
    ax.set_ylabel('赢得汽车的频率')
    return fig, ax
# %%
N = 100
np.random.seed(20)
data = np.random.rand(N)
policy1 = data<1./3
policy2 = data>=1./3
# %%
fig, ax = create_fig()
# create a figure updater
ud = UpdateFigure(ax, policy1)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=108, blit=True)
# save animation as *.mp4
anim.save(path/'goat_policy1.mp4', fps=12, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
fig, ax = create_fig()
# create a figure updater
ud = UpdateFigure(ax, policy2)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=108, blit=True)
# save animation as *.mp4
anim.save(path/'goat_policy2.mp4', fps=12, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])