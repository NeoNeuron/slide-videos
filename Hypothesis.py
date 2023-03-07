# %%
from pathlib import Path
path = Path('./videos/hypothesis_test/')
path.mkdir(parents=True, exist_ok=True)
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
plt.rcParams["font.size"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
#%%
class HtType1Error:
    def __init__(self, ax:plt.Axes, dx:float=0.05,
                 xlim=None, ylim=None, alpha=0.05,
                 fix_alpha=True, save_snapshot=False):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
            fix_alpha (bool, optional): 
                If True, fix alpha value and change epsilon (rejection boundary).
                If False, fix epsilon determined by init alpha while varying alpha. 
                Defaults to True.
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
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        self.ax.set_yticks([])

        self.norm = norm(loc=self.mean, scale=self.sigma)
        self.gauss = self.norm.pdf
        self.isf = self.norm.isf(alpha)
        self.ppf = self.norm.ppf(alpha)
        self.alpha = alpha if fix_alpha else None
        self.fix_alpha = fix_alpha

        # start ploting
        if save_snapshot:
            snapshot()

        self.line = self.ax.plot(self.x,self.gauss(self.x), color='k', zorder=1)
        self.ax.axvline(self.mean, ymax=0.95, ls='--', color='#C00000')
        if save_snapshot:
            snapshot()

        vline0 = self.ax.axvline(norm.isf(alpha, loc=self.mean, scale=self.sigma), ymax=0.8, ls='--', color='#002060')
        vline1 = self.ax.axvline(norm.ppf(alpha, loc=self.mean, scale=self.sigma), ymax=0.8, ls='--', color='#002060')
        self.vlines = [vline0, vline1]

        shade_verts = [
            self._shading_verts(self.isf, 'right'), 
            self._shading_verts(self.ppf, 'left'),
        ]
        self.shades = [
            PolyCollection(verts, fc='#0096FE', alpha=0.8, zorder=0)
            for verts in shade_verts
        ]
        [self.ax.add_collection(shade) for shade in self.shades]
        self.height = 1.2
        self.width  = 0.06
        self.offset = 0.005
        stick0, text0 = self.create_text(self.isf, 'right')
        stick1, text1 = self.create_text(self.ppf, 'left')
        self.texts = [text0, text1]
        self.sticks = [stick0, stick1]
        if save_snapshot:
            snapshot()

    def create_text(self, x, direction='right', text=r'$\frac{\alpha}{2}$'):
        if direction == 'right':
            stick, = self.ax.plot(x+np.array([self.offset, self.offset+self.width]),self.gauss(x+self.offset)/2+np.array([0., self.height]), color='k', zorder=1)
            text = self.ax.text(x+self.offset+0.005+self.width, self.height+self.gauss(x+self.offset)/2, text, ha='left')
        elif direction == 'left':
            stick, = self.ax.plot(x-np.array([self.offset, self.offset+self.width]),self.gauss(x-self.offset)/2+np.array([0., self.height]), color='k', zorder=1)
            text = self.ax.text(x-self.offset-0.005-self.width, self.height+self.gauss(x-self.offset)/2, text, ha='right')
        return stick, text

    def update_text(self, x, stick, text, direction='right'):
        if direction == 'right':
            stick.set_data(x+np.array([self.offset, self.offset+self.width]),
                           self.gauss(x+self.offset)/2+np.array([0., self.height]))
            text.set_position((x+self.offset+0.005+self.width,
                               self.height+self.gauss(x+self.offset)/2))
        elif direction == 'left':
            stick.set_data(x-np.array([self.offset, self.offset+self.width]),self.gauss(x-self.offset)/2+np.array([0., self.height]))
            text.set_position((x-self.offset-0.005-self.width, self.height+self.gauss(x-self.offset)/2))
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
        self.norm = norm(loc=self.mean, scale=self.sigma-i*self.dx)
        self.gauss = self.norm.pdf
        if self.fix_alpha:
            self.isf = self.norm.isf(self.alpha)
            self.ppf = self.norm.ppf(self.alpha)
            self.update_vlines()
        self.update_gauss()
        shade_verts = [
            self._shading_verts(self.isf, 'right'), 
            self._shading_verts(self.ppf, 'left'),
        ]
        [shade.set_verts(verts) for shade, verts in zip(self.shades, shade_verts)]
        self.update_text(self.isf, self.sticks[0], self.texts[0], 'right')
        self.update_text(self.ppf, self.sticks[1], self.texts[1], 'left')
        return self.line

class HtType2Error(HtType1Error):
    def __init__(self, ax:plt.Axes, dx:float=0.05,
                 xlim=None, ylim=None, alpha=0.05,
                 fix_alpha=True, save_snapshot=False):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
        """

        self.ax = ax
        self.ax.spines["left"].set_visible(False)
        self.dx = dx
        self.mean = 7.1
        self.sigma = 0.04
        self.th = 0.0784
        self.x = np.linspace(-0.25,0.25,500) + 7.0#self.mean

        self.ax.set_xlim(self.x[0],self.x[-1])
        self.ax.set_ylim(0, 12.0)
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        self.ax.set_yticks([])

        self.norm = norm(loc=self.mean, scale=self.sigma)
        self.norm1 = norm(loc=7, scale=self.sigma)
        self.gauss = self.norm.pdf
        self.isf = self.norm1.isf(alpha)
        self.ppf = self.norm1.ppf(alpha)
        self.alpha = alpha if fix_alpha else None
        self.fix_alpha = fix_alpha

        # start ploting
        if save_snapshot:
            snapshot()

        self.line = self.ax.plot(self.x,self.gauss(self.x), color='k', zorder=1)
        self.ax.axvline(self.mean, ymax=0.95, ls='--', color='#C00000')
        if save_snapshot:
            snapshot()

        vline0 = self.ax.axvline(self.isf, ymax=0.8, ls='--', color='#002060')
        vline1 = self.ax.axvline(self.ppf, ymax=0.8, ls='--', color='#002060')
        self.vlines = [vline0, vline1]

        # shade0 = self._shading(self.ppf, 'right')
        shade1 = self._shading_verts(self.isf, 'left')
        self.shades = [PolyCollection(shade1, fc='#F23B36', alpha=0.8, zorder=0),]
        [self.ax.add_collection(shade) for shade in self.shades]
        self.height = 1.2
        self.width  = 0.05
        self.offset = 0.010
        # stick0, text0 = self.create_text(self.ppf,'right',r'$\beta$')
        stick1, text1 = self.create_text(self.isf,'left',r'$\beta$')
        self.texts = [text1,]
        self.sticks = [stick1,]
        if save_snapshot:
            snapshot()

    # def __call__(self, i):
    #     depsilon=1.5e-4
    #     eps = 0.025-i*depsilon
    #     self.isf = self.norm1.isf(eps)
    #     self.ppf = self.norm1.ppf(eps)
    #     self.update_vlines()
    #     self.shades[0].set_verts(self._shading_verts(self.isf, 'left'),)# self._shading(self.ppf, 'left')]
    #     # self.update_text(self.isf, self.sticks[0], self.texts[0], 'right')
    #     self.update_text(self.isf, self.sticks[0], self.texts[0], 'left')
    #     return self.line

    def __call__(self, i):
        self.norm = norm(loc=self.mean, scale=self.sigma-i*self.dx)
        self.norm1 = norm(loc=7.0, scale=self.sigma-i*self.dx)
        self.gauss = self.norm.pdf
        if self.fix_alpha:
            self.isf = self.norm1.isf(self.alpha)
            self.ppf = self.norm1.ppf(self.alpha)
            self.update_vlines()
        self.update_gauss()
        # shade0 = self._shading_verts(self.ppf, 'right')
        shade1 = self._shading_verts(self.isf, 'left')
        self.shades[0].set_verts(shade1)
        # self.update_text(self.ppf, self.sticks[0], self.texts[0], 'right')
        self.update_text(self.isf, self.sticks[0], self.texts[0], 'left')
        return self.line
#%%
fig, ax = plt.subplots(1,1, figsize=(10,3),dpi=400)
prefix = 'HtErrorTy1'
counter = 1
def snapshot():
    global counter
    fig.savefig(path/f'{prefix}_snapshot_{counter}.png', dpi=300)
    counter += 1

# create a figure updater
# ud = HtType1Error(ax, dx=0.00014, save_snapshot=True)
ud = HtType1Error(ax, dx=0.00014, ylim=(0,18), fix_alpha=False)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=121, blit=True)
# save animation as *.mp4
anim.save(path/f'{prefix}_movie.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
fig, ax = plt.subplots(1,1, figsize=(10,3),dpi=400)
prefix = 'HtErrorTy2'
counter = 1
def snapshot():
    global counter
    fig.savefig(path/f'{prefix}_snapshot_{counter}.png', dpi=300)
    counter += 1

# create a figure updater
# ud = HtType2Error(ax, dx=0.00014, save_snapshot=True)
ud = HtType2Error(ax, dx=0.00014, ylim=(0,18), fix_alpha=False)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=121, blit=True)
# save animation as *.mp4
anim.save(path/f'{prefix}_movie.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%