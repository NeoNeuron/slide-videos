# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24

#%%
class UpdateFigure_scatter_hist:
    def __init__(self, data:np.ndarray, 
                 ax_main:plt.Axes, ax_right:plt.Axes, 
                 ylim:tuple=None):
        """Plot the first frame for the animation.

        Args:
            data (np.ndarray): 1-D array of number of passagers for each days
            ax_main (plt.Axes): axes of scatter plot
            ax_right (plt.Axes): axes of histogram
        """

        self.colors = dict(
            flight_init=[0,0,0,1],
            main=np.array([0,109,135,255])/255.0, #006D87
            gauss=np.array([177,90,67,255])/255.0, #B15A43
            flight_red=np.array([230,0,18,255])/255.0,
            flight_green=np.array([0,176,80,255])/255.0,
        )
        self.data = data
        self.trials = np.arange(data.shape[0])+1

        # scatter plot:
        self.line_main, = ax_main.plot([], [], 'o',
            color=self.colors['main'],
            markersize=12,
            markerfacecolor='none',
            markeredgewidth=2)

        # now determine nice limits by hand:
        ymax = np.max(np.fabs(data))
        ymin = np.min(np.fabs(data))
        xlim = (-1, data.shape[0]+1)
        if ylim is None:
            ylim = (ymin, ymax)
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        self.ax_main = ax_main

        # initialize the bins of histogram
        self.bins = 10
        self.range = list(ylim)
        counts, edges = np.histogram(data, range=self.range, bins = self.bins)
        self.binsize=edges[1]-edges[0]
        self.rects = ax_right.barh((edges[1:]+edges[:-1])/2, np.zeros_like(counts), height=self.binsize*0.96, color=self.colors['main'], alpha=0.5)

        ax_right.set_ylim(*ylim)
        ax_right.set_xlim(0,counts.max()+1)

        # fit the distribution with gaussian
        self.ax_right= ax_right
        self.last_bin=None

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0 and i<=self.data.shape[0]:
            if self.last_bin is not None:
                self.rects[self.last_bin].set_alpha(0.5)
            # update lines
            self.line_main.set_data(self.trials[:i], self.data[:i])

            # update the height of bars for histogram
            idx = int((self.data[i-1]-self.range[0])//self.binsize)
            self.rects[idx].set_width(self.rects[idx].get_width()+1)
            self.rects[idx].set_alpha(1)
            self.last_bin = idx
        elif i == self.data.shape[0]+1:
            for rect in self.rects:
                rect.set_alpha(1)
            self.ax_main.axhline(self.data.mean(), color='b',ls='--')
            self.ax_right.axhline(self.data.mean(), color='b',ls='--')
            self.ax_main.axhline(7, color='r',ls='--')
            self.ax_right.axhline(7, color='r',ls='--')
        return self.rects

# %
K = 25 # number of random tests

# generate sampling data
rv = norm.rvs(loc=7.1, scale=0.2, size=(K,), random_state=99238)

fig, ax = plt.subplots(1,2,figsize=(15,3),dpi=100, 
                       gridspec_kw={'width_ratios':[3,1], 'wspace':0.05, 'left':0.08, 'right':0.98, 'bottom':0.25, 'top':0.95})
ax[0].set_xlabel('采样次数', fontsize=25)
ax[0].set_ylabel('产量(吨/公顷)', fontsize=25)
ax[1].set_xlabel('频数', fontsize=25)
# no labels
from matplotlib.ticker import NullFormatter 
ax[1].yaxis.set_major_formatter(NullFormatter())

# create a figure updater
ud = UpdateFigure_scatter_hist(rv, ax[0], ax[1],(6.6, 7.6))
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=36, blit=True)
# save animation as *.mp4
anim.save('hypothesis_sampling.mp4', fps=6, dpi=300, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
