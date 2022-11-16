# %%
from pathlib import Path
path = Path('./lln/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
#%%
class ChangeEps:
    def __init__(self, 
        ax:plt.Axes, dx:float=0.05):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
        """

        self.ax = ax
        self.ax.spines["top"].set_visible(True)
        self.ax.spines["right"].set_visible(True)
        self.x = np.linspace(-2,2,500)
        self.depsilon = dx

        self.ax.set_xlim(-2,2)
        self.ax.set_ylim(0, 0.4)
        self.ax.set_xlabel('$X$', fontsize=20)
        self.ax.set_ylabel('$P(X)$', fontsize=20)
        self.ax.set_xticks(np.linspace(-2,2,5))
        self.line = self.ax.plot(self.x,norm.pdf(self.x, scale=0.5), color='k', zorder=1)
        self.ax.axvline(0, ymax=0.9, ls='--', color='grey')
        vline0 = self.ax.axvline(0.5, ymax=0.6, ls='--', color='#002060')
        vline1 = self.ax.axvline(-0.5, ymax=0.6, ls='--', color='#002060')
        self.vlines = [vline0, vline1]
        self.shades = self._shading(0.5)
        text0 = self.ax.text(0.5, 0.25, r'$\varepsilon$', )
        text1 = self.ax.text(-0.5, 0.25, r'$\varepsilon$', )
        self.texts = [text0, text1]

    def _shading(self,epsilon):
        art1 = self.ax.fill_between(self.x[self.x>=epsilon],  0, norm.pdf(self.x[self.x>=epsilon],  sigma=.5), fc='#C00000', alpha=0.5)
        art2 = self.ax.fill_between(self.x[self.x<=-epsilon], 0, norm.pdf(self.x[self.x<=-epsilon], sigma=.5), fc='#C00000', alpha=0.5)
        return [art1, art2]

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        self.line[0].set_data(self.x,norm.pdf(self.x, scale=0.5))
        eps = 0.5+i*self.depsilon
        [self.vlines[i].set_data(np.ones(2)*val, [0,0.6]) for i, val in enumerate((eps, -eps))]
        [shade.remove() for shade in self.shades]
        self.shades = self._shading(eps)
        [self.texts[i].set_position((val, 0.25)) for i, val in enumerate((eps, -eps))]
        return self.line
# %%
fig, ax = plt.subplots(
    1,1, figsize=(7,4),
    gridspec_kw=dict(left=0.12, right=0.95, top=0.95, bottom=0.15))

# create a figure updater
ud = ChangeEps(ax, dx=0.005)
plt.savefig(path/'test.pdf')
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=121, blit=True)
# save animation as *.mp4
anim.save(path/'change_eps.mp4', fps=60, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
class ChangeNorm:
    def __init__(self, 
        ax:plt.Axes, dx:float=0.05):
        """Plot the first frame for the animation.
        Args:
            ax (plt.Axes): axes of flight icons.
        """

        self.ax = ax
        self.ax.spines["top"].set_visible(True)
        self.ax.spines["right"].set_visible(True)
        self.x = np.linspace(-2,2,500)
        self.depsilon = dx

        self.ax.set_xlim(-2,2)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('$X$', fontsize=20)
        self.ax.set_ylabel('$P(X)$', fontsize=20)
        self.ax.set_xticks(np.linspace(-2,2,5))
        self.line, = self.ax.plot(self.x,norm.pdf(self.x, scale=0.1), color='k', zorder=1)
        self.ax.axvline(0, ymax=1.0, ls='--', color='grey')
        self.ax.axvline(0.5, ymax=0.8, ls='--', color='#002060')
        self.ax.axvline(-0.5, ymax=0.8, ls='--', color='#002060')
        self.shades = self._shading(0.5,sigma=0.1)
        self.ax.text(0.5, 0.25, r'$\varepsilon$', )
        self.ax.text(-0.5, 0.25, r'$\varepsilon$', )

    def _shading(self,epsilon,sigma):
        #画阴影部分
        art1=self.ax.fill_between(self.x[self.x>=epsilon], 0, norm.pdf(self.x[self.x>=epsilon],scale=sigma), fc='r', alpha=0.5)
        art2=self.ax.fill_between(self.x[self.x<=-epsilon], 0, norm.pdf(self.x[self.x<=-epsilon],scale=sigma), fc='r', alpha=0.5)
        return [art1, art2]

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        # clear current axis and replot
        sigma=0.1+i*0.01
        self.line.set_data(self.x,norm.pdf(self.x, scale=sigma))
        [shade.remove() for shade in self.shades]
        self.shades = self._shading(0.5,sigma)
        return self.shades
    
fig, ax = plt.subplots(
    1,1, figsize=(7,4),
    gridspec_kw=dict(left=0.12, right=0.95, top=0.95, bottom=0.15))

# create a figure updater
ud = ChangeNorm(ax, dx=0.005)
plt.savefig(path/'test.pdf')
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=121, blit=True)
# save animation as *.mp4
anim.save(path/'change_norm.mp4', fps=60, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
