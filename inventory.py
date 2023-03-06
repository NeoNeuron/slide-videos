# %%
from pathlib import Path
path = Path('./videos/poisson/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import poisson
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

class UpdateDist:
    def __init__(self, data, lam, ax, n_in_each_frame):
        #self.ax = ax
        self.n_in_each_frame = n_in_each_frame
        self.data = data
        self.cummean = np.cumsum(self.data)/np.arange(1, data.shape[0]+1)
        self.xdata = np.arange(1, data.shape[0]+1)

        # main plot:
        self.line_main, = ax.plot([], [], lw=5, 
            c='#65E8F5', clip_on=False, zorder=5, label=r'$\overline{X}$')
        self.dots, = ax.plot([], [], 'o',
            ms=8, mew=1.5, 
            mec='#353A71', mfc='#A7BED2', clip_on=True, zorder=1, label=r'$X_n$')
        ax.set_xlabel('随机次数(n)', fontsize=24)
        ax.set_ylabel(r'$X_n$', fontsize=24)
        ax.set_title(f'n = {0:<5d}    '+r'$\overline{X}$'+f' = {0:<5.2f}', fontsize=20)
        ax.legend(loc=1)

        # now determine nice limits by hand:
        ax.set_xlim((-1, 101))
        ax.set_ylim(np.array([0.1,lam*2.]))
        ax.set_yticks([lam, lam*2])
        self.ax = ax
        self.line = self.ax.axhline(lam, ls='--', color='#C00000', zorder=0)
        text = self.ax.text(0.07, 0.75, r'$\lambda$=%d'%lam, 
                     transform=self.ax.transAxes, fontsize=30, color='#C00000', zorder=20)
        text.set_bbox(dict(facecolor=[1,1,1,0.7], edgecolor='#C00000'))

    def __call__(self, i):
        if i > 0:
            idx = self.n_in_each_frame[i]
            self.dots.set_data(self.xdata[:idx], self.data[:idx])
            self.line_main.set_data(self.xdata[:idx], self.cummean[:idx])
            self.ax.set_title(f'n = {idx:<5d}    '+r'$\mathrm{\overline{X}}$'+f'= {self.cummean[idx-1]:<5.2f}', fontsize=20)
        return [self.line]

lam=10
n = 100 # number of accumulated samples
flip_results = poisson.rvs(mu=lam, size=(n,), random_state=8)
fig, ax1 = plt.subplots(1,1, figsize=(8,4), gridspec_kw={'bottom':0.2})

n_in_each_frame = (np.linspace(1,10,47)**2).astype(int)
n_in_each_frame = np.hstack(([0], n_in_each_frame))
ud = UpdateDist(flip_results, lam, ax1, n_in_each_frame)
anim = FuncAnimation(fig, ud, frames=n_in_each_frame.shape[0], blit=True)
fname = path/f"inventory_{lam:d}.mp4"
anim.save(fname, fps=12, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%