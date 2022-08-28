# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import expon
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
            markersize=8,
            markerfacecolor='none',
            markeredgewidth=2)

        # now determine nice limits by hand:
        ymax = np.max(np.fabs(data))
        ymin = np.min(np.fabs(data))
        xlim = (-1, data.shape[0]+1)
        if ylim is None:
            ylim = (np.round(ymin)-0.5, np.round(ymax)+0.5)
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        self.ax_main = ax_main

        # initialize the bins of histogram
        self.bins = int(ylim[1]-ylim[0])
        self.range = list(ylim)
        counts, edges = np.histogram(data, range=self.range, bins = self.bins)
        self.binsize=edges[1]-edges[0]
        self.rects = ax_right.barh((edges[1:]+edges[:-1])/2, np.zeros_like(counts), height=0.98, color=self.colors['main'], alpha=0.5)

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
        elif i > self.data.shape[0]:
            for rect in self.rects:
                rect.set_alpha(1)
        return self.rects

class UpdateFigure_scatter:
    def __init__(self, data:np.ndarray, 
                 ax:plt.Axes, ylim:tuple=None):
        """Plot the first frame for the animation.

        Args:
            data (np.ndarray): 1-D array of number of passagers for each days
            ax (plt.Axes): axes of scatter plot
        """

        self.colors = dict(
            main=np.array([0,109,135,255])/255.0, #006D87
        )
        self.data = data
        self.trials = np.arange(data.shape[0])+1

        # scatter plot:
        self.line_main, = ax.plot([], [], 'o',
            color=self.colors['main'],
            markersize=8,
            markerfacecolor='none',
            markeredgewidth=2)

        # now determine nice limits by hand:
        ymax = np.max(np.fabs(data))
        ymin = np.min(np.fabs(data))
        xlim = (-1, data.shape[0]+1)
        if ylim is None:
            ylim = (np.round(ymin)-0.5, np.round(ymax)+0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self.ax_main = ax

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0 and i<=self.data.shape[0]:
            # update lines
            self.line_main.set_data(self.trials[:i], self.data[:i])

        return [self.line_main]
# %%
n = 10 # number of accumulated samples
K = 200 # number of random tests

# generate sampling data
rv = expon.rvs(scale=5, size=(K,n), random_state=99239)
theta1 = np.mean(rv, axis=1, dtype=float)
theta2 = np.min(rv, axis=1)
theta3 = np.min(rv, axis=1)*n

fig, ax = plt.subplots(1,2,figsize=(10.5,7),dpi=100, 
                       gridspec_kw={'width_ratios':[3,1], 'wspace':0.1, 'left':0.12, 'right':0.98, 'bottom':0.15, 'top':0.95})
ax[0].set_xlabel('采样次数', fontsize=40)
ax[0].set_ylabel(r'$\hat{\theta}_1$', fontsize=40, usetex=True, rotation=0, ha='right', va='center')
ax[1].set_xlabel('频数', fontsize=40)
# no labels
from matplotlib.ticker import NullFormatter 
ax[1].yaxis.set_major_formatter(NullFormatter())
[axi.axhline(5, color='r',ls='--') for axi in ax]

# create a figure updater
ud = UpdateFigure_scatter_hist(theta1, ax[0], ax[1], (-0.5,32.5))
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=240, blit=True)
# save animation as *.mp4
anim.save('point_estimation_theta1.mp4', fps=24, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
# =========================================================
n = 10 # number of accumulated samples
K = 200 # number of random tests

# generate sampling data
theta1 = []
theta2 = []
theta3 = []
ns = [5, 30, 90, 200, 500]
for _n in ns:
    rv = expon.rvs(scale=5, size=(40,_n), random_state=99239)
    theta1.append(np.mean(rv, axis=1, dtype=float))
    theta2.append(np.min(rv, axis=1))
    theta3.append(np.min(rv, axis=1)*_n)
theta1 = np.hstack(theta1)
theta2 = np.hstack(theta2)
theta3 = np.hstack(theta3)

fig, ax = plt.subplots(1,1,figsize=(10.5,7),dpi=100, 
                       gridspec_kw={'left':0.12, 'right':0.98, 'bottom':0.15, 'top':0.95})
ax.set_xlabel('采样次数', fontsize=40)
ax.set_ylabel(r'$\hat{\theta}_2$', fontsize=40, usetex=True, rotation=0, ha='right', va='center')
ax.set_xticks([0,40,80,120,160,200])
ax.axhline(5, color='r',ls='--')
for i in range(4):
    ax.axvline(40*(i+1), color='gray',ls='--')
for i in range(5):
    ax.text(40*i+20, 28, f'n={ns[i]:d}', ha='center', va='center', fontsize=30)
ud = UpdateFigure_scatter(theta3, ax, (-0.5,30))
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=240, blit=True)
# save animation as *.mp4
anim.save('point_estimation_theta1.mp4', fps=24, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
