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
        self.line = self.ax.plot(self.x,self._gauss(0, 0.5, self.x), color='k', zorder=1)
        self.ax.axvline(0, ymax=0.9, ls='--', color='#C00000')
        vline0 = self.ax.axvline(0.5, ymax=0.6, ls='--', color='#002060')
        vline1 = self.ax.axvline(-0.5, ymax=0.6, ls='--', color='#002060')
        self.vlines = [vline0, vline1]
        self.shades = self._shading(0.5)
        text0 = self.ax.text(0.5, 0.25, r'$\varepsilon$', )
        text1 = self.ax.text(-0.5, 0.25, r'$\varepsilon$', )
        self.texts = [text0, text1]

    @staticmethod
    def _gauss(mean, sigma, x):
        return np.exp(-(x-mean)**2/(2*sigma**2))/2/np.pi/sigma

    def _shading(self,epsilon):
        art1 = self.ax.fill_between(self.x[self.x>=epsilon], 0,  self._gauss(0, 0.5, self.x[self.x>=epsilon]), fc='#1F77B4', alpha=0.5)
        art2 = self.ax.fill_between(self.x[self.x<=-epsilon], 0, self._gauss(0, 0.5, self.x[self.x<=-epsilon]), fc='#1F77B4', alpha=0.5)
        return [art1, art2]

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        self.line[0].set_data(self.x,self._gauss(0, 0.5, self.x))
        eps = 0.5+i*self.depsilon
        [self.vlines[i].set_data(np.ones(2)*val, [0,0.6]) for i, val in enumerate((eps, -eps))]
        [shade.remove() for shade in self.shades]
        self.shades = self._shading(eps)
        [self.texts[i].set_position((val, 0.25)) for i, val in enumerate((eps, -eps))]
        return self.line
# %%
fig, ax = plt.subplots(1,1, figsize=(8,4),dpi=200)

# create a figure updater
ud = UpdateFigure(ax, dx=0.005)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=121, blit=True)
# save animation as *.mp4
anim.save('chebyshev_movie.mp4', fps=60, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%