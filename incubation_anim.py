#%%
from pathlib import Path
path = Path('./function_of_random_variables/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'Microsoft YaHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["axes.labelsize"] = 45
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.left"] = False
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
#%%
def axesArrow(ax, xlim, ylim):
    ax.scatter(xlim[-1], ylim[0], s=180, color='k', marker='>', ).set_clip_on(False)
    for axis in ['bottom',]:
        ax.spines[axis].set_linewidth(5)
    return ax

def configAxes(ax, xlim, ylim):
    ax = axesArrow(ax, xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([0,2,5])
    ax.set_yticks([])
    return ax


class incubation_motion:
    def __init__(self, 
        ax:plt.Axes, fps:int=24):
        self.ax = ax
        self.fps = fps
        self.shade = None
        self.tickers = [None, None, None]
        self.line, = self.ax.plot([],[])

    def __call__(self, i):
        if i == self.fps*0.5:
            self.shade = self.ax.fill_between([3,5], 0, 1., color='#FFCD00', alpha=1, zorder=-1)
            savefig(fig)
        elif i == self.fps*1:
            self.tickers[0] = self.add_down_ticker(0, '感染开始', color='r', fontsize=25)
            savefig(fig)
        elif i == self.fps*1.5:
            self.tickers[1] = self.add_up_ticker(5, '潜伏期结束', y=1.5, color='r', fontsize=25)
            savefig(fig)
        elif i == self.fps*2:
            self.tickers[2] = self.add_up_ticker(3, '传染期开始', y=1.5, color='g', fontsize=25)
            savefig(fig)
        elif i >= self.fps*2.5:
            self.shade.remove()
            new_x = 5 - (i - self.fps*2.5) * 1.0/self.fps
            self.shade = self.ax.fill_between([new_x-2,new_x], 0, 1., color='#FFCD00', alpha=1, zorder=-1)
            self.move_ticker(self.tickers[1], new_x)
            if new_x - 2 > 0:
                self.move_ticker(self.tickers[2], new_x-2)
            else:
                self.tickers[2][0].set_ydata([2.5,])
                self.tickers[2][1].set_y(3.5)
                self.move_ticker(self.tickers[2], 0)
        return [self.line]

    def add_up_ticker(self, x, text, y=3, color='r', fontsize=20):
        arrow, = self.ax.plot([x,], [y,], 'v', color=color, ms=20, clip_on=False)
        text = self.ax.text(x, y+1, text, color=color, ha='center', va='center', fontsize=fontsize, clip_on=False)
        return [arrow, text]

    def add_down_ticker(self, x, text, color='r', fontsize=20):
        arrow, = self.ax.plot([x,], [-1.,], '^', color=color, ms=20, clip_on=False)
        text = self.ax.text(x, -2.0, text, color=color, ha='center', va='center', fontsize=fontsize, clip_on=False)
        return [arrow, text]

    @staticmethod
    def move_ticker(ticker, x):
        ticker[0].set_xdata([x,])
        ticker[1].set_x(x)
        return ticker

counter = 1

def savefig(fig):
    global counter
    fig.savefig(path/f'incubation_{counter}.pdf')
    counter += 1

fig, ax = plt.subplots(1,1, figsize=(10,4), gridspec_kw=dict(left=0.02, right=0.94, bottom=0.35) )
# draw axes arrow
xlim=[-3.0, 6.5]
ylim=[0, 4.1]
configAxes(ax, xlim, ylim)
duration = 6.4
fps = 48
nframe = int(duration*fps + 1)
ax.fill_between([-3,0], 0, 1, color='w', alpha=0.8, zorder=0)
ax.text(1, -0.25, '天数', color='k', ha='center', va='center', fontsize=40, transform=ax.transAxes, clip_on=False)
# create a figure updater
ud = incubation_motion(ax, fps)
savefig(fig)
anim = FuncAnimation(fig, ud, frames=nframe, blit=True)
fig.savefig(path/'test.pdf')
anim.save(path/'demo.mp4', fps=fps, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
