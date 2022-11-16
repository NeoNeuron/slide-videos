# %%
from pathlib import Path
path = Path('normal_distribution/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
#%%
class UpdateFigure:
    def __init__(self, 
        ax:plt.Axes, dx:float=0.05):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
        """

        self.ax = ax
        self.ax.spines["top"].set_visible(True)
        self.ax.spines["right"].set_visible(True)
        self.x = np.linspace(-6,16,1000)
        self.depsilon = dx

        self.ax.set_xlim(-6,16)
        self.ax.set_ylim(0, 0.5)
        self.line0 = self.ax.plot(self.x,norm.pdf(self.x, 0, 1), color='k', zorder=1, label='$N(0,1)$')
        self.line1 = self.ax.plot(self.x,norm.pdf(self.x, 10, 2), color='k', zorder=1, label='$N(10,2)$')
        self.vline0 = self.ax.axvline(0, ymax=2*norm.pdf(0, 0, 1), ls='--', color='#C00000')
        self.vline1 = self.ax.axvline(10, ymax=2*norm.pdf(10, 10, 2), ls='--', color='#C00000')
        self.vline00 = self.ax.axvline(1, ymax=2*norm.pdf(1, 0, 1), color='#C00000')
        self.vline11 = self.ax.axvline(12, ymax=2*norm.pdf(12, 10, 2), color='#C00000')
        self.shades = self._shading(10, 2)
        self.ax.fill_between(self.x[self.x<=1], 0, norm.pdf(self.x[self.x<=1],0,1), fc='#1F77B4', alpha=0.5)
        self.ax.fill_between(self.x[self.x>=1], 0, norm.pdf(self.x[self.x>=1],0,1), fc='#C00000', alpha=0.5)
        # text label for standard normal
        self.text0 = self.ax.text(-4,0.2,r'$N(0,1)$', ha='center', va='center', color='grey')
        self.ax.plot([-3.1, -1.6],[0.17,norm.pdf(-1.6)], color='grey')
        # text label for varying normal
        self.text1 = self.ax.text(10,0.42,r'$N(10,4)$', va='bottom', ha='center')
        self.ax.set_ylabel('概率密度')

    def _shading(self, mu, sigma):
        art1 = self.ax.fill_between(self.x[self.x<=mu+sigma], 0, norm.pdf(self.x[self.x<=mu+sigma],mu,sigma), fc='#1F77B4', alpha=0.5)
        art2 = self.ax.fill_between(self.x[self.x>=mu+sigma], 0, norm.pdf(self.x[self.x>=mu+sigma],mu,sigma), fc='#C00000', alpha=0.5)
        return [art1, art2]

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        eps = 10-i*self.depsilon
        if eps>=0:
            self.line1[0].set_data(self.x,norm.pdf(self.x,eps,2))
            self.vline1.set_data([eps,eps],[0,2*norm.pdf(eps, eps, 2)])
            self.vline11.set_data([eps+2,eps+2],[0,2*norm.pdf(eps+2, eps, 2)])
            [shade.remove() for shade in self.shades]
            self.shades = self._shading(eps,2)
            self.text1.set_text('$N($'+str(round(eps,1))+'$,4)$')
            self.text1.set_x(eps)
        else:
            epssigma = 2-np.maximum(i-208, 0)*self.depsilon/5
            if epssigma < 1:
                epssigma = 1
            self.line1[0].set_data(self.x,norm.pdf(self.x,0,epssigma))
            self.vline1.set_data([0,0],[0,2*norm.pdf(0, 0, epssigma)])
            self.vline11.set_data([epssigma,epssigma],[0,2*norm.pdf(epssigma, 0, epssigma)])
            [shade.remove() for shade in self.shades]
            self.shades = self._shading(0,epssigma)
            self.text1.set_text('$N(0,$'+str(round(epssigma**2,1))+'$)$')
        return self.line1
fig, ax = plt.subplots(
    1,1, figsize=(8,3),
    gridspec_kw=dict(left=0.1, right=0.95, top=0.95, bottom=0.1))

# create a figure updater
ud = UpdateFigure(ax, dx=0.05)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=321, blit=True)
# save animation as *.mp4
anim.save(path/'standardize_norm.mp4', fps=40, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%