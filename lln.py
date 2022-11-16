# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
from scipy.ndimage import gaussian_filter1d
#%%
class UpdateFigure:
    def __init__(self, 
        ax:plt.Axes, data:np.ndarray, stride:int=1):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
            data (np.ndarray): 1-D array of number of passagers for each days
        """

        self.ax = ax

        self.ax.set_xlim(1,data.shape[1])
        self.ax.set_ylim(-1,1)
        self.ax.set_yticklabels([])
        self.ax.set_xlabel('n', fontsize=24)
        self.ax.set_ylabel(r'$\hat{\theta}$', fontsize=24)
        # self.line_symbolic =self.ax.plot(np.arange(data.shape[1]),dx*np.sqrt(np.arange(data.shape[1])), ls='--', color='k')
        self.line =self.ax.plot([],[], zorder=1, color='#55B046')
        self.xc = 0.0
        self.yc = 0.0
        self.dot, =self.ax.plot([0],[0], 'o', color='#FFF600', ms=2, zorder=1)
        self.stride = stride
        # Draw all samples
        data_all = np.cumsum(data, axis=1)/np.arange(1, data.shape[1]+1)
        self.data_resample = data_all[:, ::stride]
        self.data = self.data_resample[0]
        self.x_data = np.arange(0, data.shape[1], stride)
        self.x_data = self.x_data
        [self.ax.plot(self.x_data, data_single, color='grey', alpha=0.4, lw=0.2, zorder=0) for data_single in self.data_resample]
        # self.draw_distribution()

    def draw_distribution(self):
        factor = 10
        for sampling_point in (3, 20, 40, 70, 110, 160,):
            data_buff = self.data_resample[:,sampling_point]
            val_range = (data_buff.min()*2, data_buff.max()*2)
            counts, bins = np.histogram(self.data_resample[:,sampling_point], bins=100, range=val_range, density=True)
            self.ax.plot(gaussian_filter1d(counts,sigma=6)*factor+ self.x_data[sampling_point], bins[1:], 'r')
            self.ax.fill_betweenx(
                bins[1:], np.ones_like(bins[1:], dtype=float)*self.x_data[sampling_point],
                gaussian_filter1d(counts,sigma=6)*factor+ self.x_data[sampling_point], color='r', alpha=0.3)

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i < self.data.shape[0]:
            # update brownian particle
            self.dot.set_data(self.x_data[i], self.data[i])
            # update curve
            self.line[0].set_data(self.x_data[:i+1], self.data[:i+1])
        if i == self.data.shape[0]+24:
            self.draw_distribution()

        return self.line
# %%
fig, ax = plt.subplots(1,1, figsize=(10,4),)

np.random.seed(2022)
data = np.random.rand(400, 2000)*2-1

# create a figure updater
ud = UpdateFigure(ax, data, 10)
plt.tight_layout()
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=250, blit=True)
# save animation as *.mp4
anim.save('lln_movie.mp4', fps=48, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%