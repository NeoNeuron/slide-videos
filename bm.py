# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
#%%
class UpdateFigure:
    def __init__(self, 
        ax:plt.Axes, data:np.ndarray, dx:float=0.05):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
            data (np.ndarray): 1-D array of number of passagers for each days
        """

        self.ax = ax
        self.ax[0].set_xlim(-2,2)
        self.ax[0].set_ylim(-2,2)
        # self.ax[0].axis('off')
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])
        self.ax[0].spines["top"].set_visible(True)
        self.ax[0].spines["right"].set_visible(True)
        self.dot, =self.ax[0].plot([0],[0], 'o', color='#FFF600', ms=2, zorder=1)

        self.ax[1].set_xlim(0,data.shape[1])
        self.ax[1].set_ylim(0,2)
        self.ax[1].set_xlabel('时间')
        self.ax[1].set_ylabel('距离')
        self.data = data[0]
        self.line_symbolic =self.ax[1].plot(np.arange(data.shape[1]),dx*np.sqrt(np.arange(data.shape[1])), ls='--', color='k')
        self.line, =self.ax[1].plot([],[], zorder=1, color='#B72C31')
        self.xc = 0.0
        self.yc = 0.0
        self.dx = dx
        # Draw all samples
        dx_all = self.dx*np.cos(data*np.pi*2)
        dy_all = self.dx*np.sin(data*np.pi*2)
        x_all = np.hstack((np.zeros((data.shape[0], 1)),np.cumsum(dx_all, axis=1)))
        y_all = np.hstack((np.zeros((data.shape[0], 1)),np.cumsum(dy_all, axis=1)))
        [self.ax[0].plot(x, y, color='grey', alpha=0.5, lw=0.5, zorder=0) for x, y in zip(x_all, y_all)]
        dis_all = self.dis2o(x_all, y_all)
        [self.ax[1].plot(dis, color='grey', alpha=0.5, lw=0.4, zorder=0) for dis in dis_all]



    @staticmethod
    def dis2o(x, y):
        return np.sqrt(x**2+y**2)

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0 and i < self.data.shape[0]:
            xnew = self.xc + self.dx*np.cos(self.data[i]*np.pi*2)
            ynew = self.yc + self.dx*np.sin(self.data[i]*np.pi*2)
            # update brownian motion
            self.ax[0].plot([self.xc, xnew], [self.yc, ynew], color='#B72C31',lw=0.5, zorder=0)
            self.xc = xnew
            self.yc = ynew
            # update brownian particle
            self.dot.set_data(self.xc, self.yc)
            # update curve
            xdata, ydata = self.line.get_data()
            if len(xdata) == 0:
                xdata = [0]
                ydata = [0]
            else:
                xdata = np.append(xdata, i) 
                ydata = np.append(ydata, self.dis2o(self.xc, self.yc)) 
            self.line.set_data(xdata, ydata)
        return self.line_symbolic
# %%
fig, ax = plt.subplots(1,2, figsize=(12,3),dpi=200, gridspec_kw=dict(width_ratios=[1,4], left=0.05, right=0.95, bottom=0.2))

np.random.seed(2022)
data = np.random.rand(100, 400)

# create a figure updater
ud = UpdateFigure(ax, data, dx=0.05)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=399, blit=True)
# save animation as *.mp4
anim.save('bm_movie.mp4', fps=48, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%