# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
        ax:plt.Axes, ratio:np.ndarray):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
            data (np.ndarray): 1-D array of number of passagers for each days
        """

        self.ax = ax
        self.ratio = ratio
        self.cov = np.array([[1,0.95],[0.95,1]])
        self.data = self.get_data(np.eye(2)).T
        self.pca = PCA(n_components=2)
        self.score = self.pca.fit_transform(self.data)
        self.score /= self.pca.singular_values_/np.sqrt(10000)

        self.line = self.ax.scatter(self.data[:,0], self.data[:,1], s=10, alpha=.3, edgecolors='None')
            # facecolor='#B4C7E7', edgecolor='#2E5597')
        # self.line.set_color(plt.cm.rainbow(data[0]))
        self.ax.axis('scaled')
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-3.5, 3.5)
        self.ax.axis('off')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

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
        X = np.random.multivariate_normal(mean, cov_matrix, size=10000)
        return X.T

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i < self.ratio.shape[0]:
            # regenerate data
            score = self.score.copy()
            score[:,0] *= np.sqrt(self.ratio[i, 0])
            score[:,1] *= np.sqrt(self.ratio[i, 1])
            new_X = self.pca.inverse_transform(score)
            # data = self.get_data(self.cov)
            self.line.set_offsets(new_X)
            # self.line.set_color(plt.cm.rainbow(data[0]))
            # initialize text
            self.ax.set_title(f'{np.corrcoef(new_X.T)[0,1]:^5.2f}', fontsize=40)
        return [self.line,]
# %%
fig, ax = plt.subplots(1,1, figsize=(6,6),dpi=400)

np.random.seed(2022)
nframes = 240
ratio = np.linspace(0,1, nframes)
ratio = np.vstack((ratio, np.flip(ratio))).T

# create a figure updater
ud = UpdateFigure(ax, ratio)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save('corr_v3_movie.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%