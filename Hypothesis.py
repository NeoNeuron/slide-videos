# %%
from functools import partial
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
        self.ax.spines["left"].set_visible(False)
        self.dx = dx
        self.mean = 7.0
        self.sigma = 0.05
        self.th = 0.0784
        self.x = np.linspace(-0.25,0.25,500) + self.mean

        self.ax.set_xlim(self.x[0],self.x[-1])
        self.ax.set_ylim(0, 6.0)
        self.gauss = partial(self._gauss, mean=self.mean, sigma=self.sigma)
        self.line = self.ax.plot(self.x,self.gauss(self.x), color='k', zorder=1)
        self.ax.axvline(self.mean, ymax=0.95, ls='--', color='#C00000')

        vline0 = self.ax.axvline(self.mean+self.th, ymax=0.7, ls='--', color='#002060')
        vline1 = self.ax.axvline(self.mean-self.th, ymax=0.7, ls='--', color='#002060')
        self.vlines = [vline0, vline1]

        shade0 = self._shading(self.mean+self.th, 'right')
        shade1 = self._shading(self.mean-self.th, 'left')
        self.shades = [shade0, shade1]
        stick0, text0 = self.create_text(self.mean+self.th, 'right')
        stick1, text1 = self.create_text(self.mean-self.th, 'left')
        self.texts = [text0, text1]
        self.sticks = [stick0, stick1]
        self.ax.set_yticks([])

    def create_text(self, x, direction='right'):
        height = 1.2
        width = 0.06
        offset = 0.005
        if direction == 'right':
            stick, = self.ax.plot(x+np.array([offset, offset+width]),self.gauss(x+offset)/2+np.array([0., height]), color='k', zorder=1)
            text = self.ax.text(x+offset+0.02+width, height+self.gauss(x+offset)/2, r'$\frac{\alpha}{2}$', ha='left')
        elif direction == 'left':
            stick, = self.ax.plot(x-np.array([offset, offset+width]),self.gauss(x-offset)/2+np.array([0., height]), color='k', zorder=1)
            text = self.ax.text(x-offset-0.02-width, height+self.gauss(x-offset)/2, r'$\frac{\alpha}{2}$', ha='right')
        return stick, text

    def update_text(self, x, stick, text, direction='right'):
        height = 1.2
        width = 0.06
        offset = 0.005
        if direction == 'right':
            stick.set_data(x+np.array([offset, offset+width]),self.gauss(x+offset)/2+np.array([0., height]))
            text.set_position((x+offset+0.02+width, height+self.gauss(x+offset)/2))
        elif direction == 'left':
            stick.set_data(x-np.array([offset, offset+width]),self.gauss(x-offset)/2+np.array([0., height]))
            text.set_position((x-offset-0.02-width, height+self.gauss(x-offset)/2))
        return stick, text

    def update_gauss(self):
        self.line[0].set_data(self.x,self.gauss(self.x))

    @staticmethod
    def _gauss(x, mean, sigma):
        return np.exp(-(x-mean)**2/(2*sigma**2))/2/np.pi/sigma

    def _shading(self, x, direction='right'):
        if direction == 'right':
            art = self.ax.fill_between(self.x[self.x>=x], 0,  self.gauss(self.x[self.x>=x]), fc='#1F77B4', alpha=0.5)
        if direction == 'left':
            art = self.ax.fill_between(self.x[self.x<=x], 0, self.gauss(self.x[self.x<=x]), fc='#1F77B4', alpha=0.5)
        return art

    # def __call__(self, i):
    #     # This way the plot can continuously run and we just keep
    #     # watching new realizations of the process
    #     eps = 0.5+i*self.depsilon
    #     [self.vlines[i].set_data(np.ones(2)*val, [0,0.6]) for i, val in enumerate((eps, -eps))]
    #     [shade.remove() for shade in self.shades]
    #     self.shades = self._shading(eps)
    #     self.update_text(eps, self.sticks[0], self.texts[0], 'r')
    #     self.update_text(-eps, self.sticks[1], self.texts[1], 'l')
    #     return self.line

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        self.gauss = partial(self._gauss, mean=self.mean, sigma=self.sigma-i*self.dx)
        self.update_gauss()
        # eps = 0.5+i*self.depsilon
        # [self.vlines[i].set_data(np.ones(2)*val, [0,0.6]) for i, val in enumerate((eps, -eps))]
        [shade.remove() for shade in self.shades]
        shade0 = self._shading(self.mean+self.th, 'right')
        shade1 = self._shading(self.mean-self.th, 'left')
        self.shades = [shade0, shade1]
        self.update_text(self.mean+self.th, self.sticks[0], self.texts[0], 'right')
        self.update_text(self.mean-self.th, self.sticks[1], self.texts[1], 'left')
        return self.line
# %%
fig, ax = plt.subplots(1,1, figsize=(10,3),dpi=400)

# create a figure updater
ud = UpdateFigure(ax, dx=0.00014)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=121, blit=True)
# save animation as *.mp4
anim.save('hypothesis_movie.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%