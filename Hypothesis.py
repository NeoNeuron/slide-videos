# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
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
prefix = 'Hypothesis_movie'
class UpdateFigure:
    def __init__(self, 
        ax:plt.Axes, dx:float=0.05):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
        """

        self.ax = ax
        self.ax.spines["left"].set_visible(False)
        self.dx = dx
        self.mean = 7.0
        self.sigma = 0.04
        self.th = 0.0784
        self.x = np.linspace(-0.25,0.25,1000) + self.mean

        self.ax.set_xlim(self.x[0],self.x[-1])
        self.ax.set_ylim(0, 12.0)
        self.norm = norm(loc=self.mean, scale=self.sigma)
        self.gauss = self.norm.pdf
        self.isf = self.norm.isf(0.025)
        self.ppf = self.norm.ppf(0.025)
        self.ax.set_yticks([])

        # start ploting
        plt.savefig(prefix+'_step1.pdf')
        self.line = self.ax.plot(self.x,self.gauss(self.x), color='k', zorder=1)
        self.ax.axvline(self.mean, ymax=0.95, ls='--', color='#C00000')
        plt.savefig(prefix+'_step2.pdf')

        vline0 = self.ax.axvline(norm.isf(0.025, loc=self.mean, scale=self.sigma), ymax=0.8, ls='--', color='#002060')
        vline1 = self.ax.axvline(norm.ppf(0.025, loc=self.mean, scale=self.sigma), ymax=0.8, ls='--', color='#002060')
        self.vlines = [vline0, vline1]
        # plt.savefig(prefix+'_step3.pdf')

        shade_verts = [
            self._shading_verts(self.isf, 'right'), 
            self._shading_verts(self.ppf, 'left'),
        ]
        self.shades = [
            PolyCollection(verts, fc='#0096FE', alpha=0.8, zorder=0)
            for verts in shade_verts
        ]
        [self.ax.add_collection(shade) for shade in self.shades]
        stick0, text0 = self.create_text(self.isf, 'right')
        stick1, text1 = self.create_text(self.ppf, 'left')
        self.texts = [text0, text1]
        self.sticks = [stick0, stick1]
        plt.savefig(prefix+'_step3.pdf')

    def create_text(self, x, direction='right'):
        height = 1.2
        width = 0.06
        offset = 0.005
        if direction == 'right':
            stick, = self.ax.plot(x+np.array([offset, offset+width]),self.gauss(x+offset)/2+np.array([0., height]), color='k', zorder=1)
            text = self.ax.text(x+offset+0.005+width, height+self.gauss(x+offset)/2, r'$\frac{\alpha}{2}$', ha='left')
        elif direction == 'left':
            stick, = self.ax.plot(x-np.array([offset, offset+width]),self.gauss(x-offset)/2+np.array([0., height]), color='k', zorder=1)
            text = self.ax.text(x-offset-0.005-width, height+self.gauss(x-offset)/2, r'$\frac{\alpha}{2}$', ha='right')
        return stick, text

    def update_text(self, x, stick, text, direction='right'):
        height = 1.2
        width = 0.06
        offset = 0.005
        if direction == 'right':
            stick.set_data(x+np.array([offset, offset+width]),self.gauss(x+offset)/2+np.array([0., height]))
            text.set_position((x+offset+0.005+width, height+self.gauss(x+offset)/2))
        elif direction == 'left':
            stick.set_data(x-np.array([offset, offset+width]),self.gauss(x-offset)/2+np.array([0., height]))
            text.set_position((x-offset-0.005-width, height+self.gauss(x-offset)/2))
        return stick, text

    def update_gauss(self):
        self.line[0].set_data(self.x,self.gauss(self.x))

    def update_vlines(self):
        [self.vlines[i].set_data(np.ones(2)*val, [0,0.8]) for i, val in enumerate((self.isf, self.ppf))]

    def _shading_verts(self, x, direction='right'):
        if direction == 'right':
            _x = self.x[self.x>=x]
            _y = self.gauss(self.x[self.x>=x])
        elif direction == 'left':
            _x = self.x[self.x<=x]
            _y = self.gauss(self.x[self.x<=x])
        verts = [[_xx, _yy] for _xx, _yy in zip(_x, _y)]
        verts += [[_x[-1], 0],[_x[0], 0]]
        return np.array([verts])

    # def __call__(self, i):
    #     # This way the plot can continuously run and we just keep
    #     # watching new realizations of the process
    #     depsilon=1.5e-4
    #     eps = 0.025-i*depsilon
    #     self.isf = self.norm.isf(eps)
    #     self.ppf = self.norm.ppf(eps)
    #     self.update_vlines()
    #     shade_verts = [
    #         self._shading_verts(self.isf, 'right'), 
    #         self._shading_verts(self.ppf, 'left' ),
    #     ]
    #     [shade.set_verts(verts) for shade, verts in zip(self.shades, shade_verts)]
    #     self.update_text(self.isf, self.sticks[0], self.texts[0], 'right')
    #     self.update_text(self.ppf, self.sticks[1], self.texts[1], 'left')
    #     return self.line

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        self.norm = norm(loc=self.mean, scale=self.sigma-i*self.dx)
        self.gauss = self.norm.pdf
        self.isf = self.norm.isf(0.025)
        self.ppf = self.norm.ppf(0.025)
        self.update_gauss()
        self.update_vlines()
        shade_verts = [
            self._shading_verts(self.isf, 'right'), 
            self._shading_verts(self.ppf, 'left'),
        ]
        [shade.set_verts(verts) for shade, verts in zip(self.shades, shade_verts)]
        self.update_text(self.isf, self.sticks[0], self.texts[0], 'right')
        self.update_text(self.ppf, self.sticks[1], self.texts[1], 'left')
        return self.line
fig, ax = plt.subplots(1,1, figsize=(10,3),dpi=400)

# create a figure updater
ud = UpdateFigure(ax, dx=0.00014)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=121, blit=True)
# save animation as *.mp4
anim.save('hypothesis_movie.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%