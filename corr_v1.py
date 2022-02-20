# %%
import numpy as np
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
        ax:plt.Axes, rho:np.ndarray):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
            data (np.ndarray): 1-D array of number of passagers for each days
        """

        self.ax = ax
        self.rho = rho
        self.cov = np.ones((2,2))
        data = self.get_data(self.cov)
        self.line = self.ax.scatter(data[0], data[1], s=10, alpha=.5, )
            # facecolor='#B4C7E7', edgecolor='#2E5597')
        self.line.set_color(plt.cm.rainbow(data[0]))
        self.ax.axis('scaled')
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-3.5, 3.5)
        # self.ax.axis('off')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.plot((1), (-3.5), ls="", marker=">", ms=10, color="k",
                transform=self.ax.get_yaxis_transform(), clip_on=False)
        self.ax.plot((-3.5), (1), ls="", marker="^", ms=10, color="k",
                transform=self.ax.get_xaxis_transform(), clip_on=False)

        # initialize text
        self.ax.set_title(f'{1:^5.2f}', fontsize=40)

    @staticmethod
    def get_data(cov_matrix):
        """
        Returns a matrix of 1000 samples from a bivariate, zero-mean Gaussian

        Args:
        cov_matrix (numpy array of floats) : desired covariance matrix

        Returns:
        (numpy array of floats)            : samples from the bivariate Gaussian,
                                            with each column corresponding to an
                                            individual sample bivariate Gaussian.
        """

        mean = np.array([0, 0])
        X = np.random.multivariate_normal(mean, cov_matrix, size=1000)
        return X.T

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i < self.rho.shape[0]:
            self.cov[0,1] = self.cov[1,0] = self.rho[i]
            data = self.get_data(self.cov)
            self.line.set_offsets(data.T)
            self.line.set_color(plt.cm.rainbow(data[0]))
            # initialize text
            self.ax.set_title(f'{self.rho[i]:^5.2f}', fontsize=40)
        return [self.line,]
# %%
fig, ax = plt.subplots(1,1, figsize=(6,6),dpi=400)

np.random.seed(2022)
nframes = 37
rho = np.linspace(1,-1, nframes)

# create a figure updater
ud = UpdateFigure(ax, rho)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save('corr_v1_movie.mp4', fps=2, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%