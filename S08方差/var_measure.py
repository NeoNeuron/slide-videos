# %%
from init import *
from matplotlib.ticker import NullFormatter 
from scipy.stats import norm
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24

#%%
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
        self.line_main.set_clip_on(False)

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
# =========================================================
# generate sampling data
theta1 = []
theta2 = []
theta3 = []
batch_size = [1, 5]
# n_samples = [240, 80, 24]
n_samples = [150, 150]
for _n, _bs in zip(n_samples, batch_size):
    rv = norm.rvs(size=(_n,_bs), random_state=9923)
    theta1.append(np.mean(rv, axis=1, dtype=float))
    rv = norm.rvs(size=(_n,_bs), random_state=923)
    theta2.append(np.median(rv, axis=1))
    if _bs > 1:
        # factor = (1/(_bs-np.arange(int(_bs/2)))).sum()
        factor = 1
        theta3.append(np.median(rv, axis=1)/factor)
    else:
        theta3.append(np.mean(rv, axis=1, dtype=float))
theta1 = np.hstack(theta1)
theta2 = np.hstack(theta2)
theta3 = np.hstack(theta3)
thetas = [theta1, theta3]
ylabels = [r'$\hat{\theta}_1$', r'$\hat{\theta}_2$']
fname_tags = ['mean', 'median_fix']
ylims = [(-5,5), (-5,5)]
def gen_activation2(theta, ylabel, fname_tag, ylim):
    fig, ax = plt.subplots(1,1,figsize=(10.5,7),dpi=100, 
                       gridspec_kw={'left':0.12, 'right':0.98, 
                                    'bottom':0.15, 'top':0.95})
    ax.set_xlabel('采样序号', fontsize=40)
    ax.set_ylabel(ylabel, fontsize=40, usetex=True, rotation=0, ha='right', va='center')
    ax.set_yticks(np.arange(-4,5,2))
    ax.set_yticklabels([r'$\mu-4\sigma$', r'$\mu-2\sigma$', r'$\mu$', r'$\mu+2\sigma$', r'$\mu+4\sigma$'])
    ax.axhline(5, color='r',ls='--')
    for i in range(len(batch_size)):
        ax.axvline(np.cumsum(n_samples)[i], color='gray',ls='--')
    for i in range(len(batch_size)):
        ax.text(np.cumsum(n_samples)[i] - n_samples[i]/2, 4.2, f'n={batch_size[i]:d}', ha='center', va='center', fontsize=30)
    ud = UpdateFigure_scatter(theta, ax, ylim)
    # user FuncAnimation to generate frames of animation
    anim = FuncAnimation(fig, ud, frames=360, blit=True)
    # save animation as *.mp4
    anim.save(path/f'point_estimation_var_n_{fname_tag}.mp4', fps=48, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

for theta, ylabel, fname_tag, ylim in zip(thetas, ylabels, fname_tags, ylims):
    print(theta.max())
    print(ylabel)
    print(fname_tag)
    print(ylim)
    print('-------')
    gen_activation2(theta, ylabel, fname_tag, ylim)
# %%
