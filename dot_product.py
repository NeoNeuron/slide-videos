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
        ax:plt.Axes, dtheta:np.ndarray):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
        """

        self.ax = ax

        self.arrows = np.array([
            [1.5, 0],
            [1, 2],
        ]).astype(float)
        self.dtheta = dtheta
        self._config_axis()
        self._draw_arrows()
        self._shading()

    def _config_axis(self):
        self.ax.set_xlim(-4,4)
        self.ax.set_ylim(-1,3)
        self.ax.axis('scaled')
        self.ax.set_xticks(np.arange(-4,5))
        self.ax.set_yticks(np.arange(-1,4))
        self.ax.grid(ls='--', alpha=0.5,zorder=0)
        self.ax.set_title(r'$\vec{x}\cdot \vec{y}$ = %+6.3f'% (self.arrows[0,0]*self.arrows[1,0]), fontsize=40)

    def _draw_arrows(self):
        self.ax.arrow(0,0,*self.arrows[0], width=.05, color='b', ec="None", length_includes_head=True, zorder=1)
        self.ax.arrow(0,0,*self.arrows[1], width=.05, color='r', ec="None", length_includes_head=True, zorder=1)
        self.ax.arrow(0,0,self.arrows[1,0], self.arrows[0,1], width=.05, color='g', ec="None", length_includes_head=True, zorder=2)
        artist = self.ax.arrow(0,0,self.arrows[0,0]*self.arrows[1,0], self.arrows[0,1], width=.05, color='cyan', ec="None", length_includes_head=True, zorder=1)
        return artist

    def _shading(self,):
        self.ax.fill_between([0,self.arrows[1][0]], 0, [0,self.arrows[1][1]], fc='r', alpha=0.3)

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        # Update arrow 1
        theta = np.arctan(self.arrows[1, 1]/self.arrows[1,0])
        arrow_len = np.sqrt((self.arrows[1]**2).sum())
        if theta < 0:
            theta += np.pi
        self.arrows[1,0] = arrow_len*np.cos(theta + self.dtheta[i])
        self.arrows[1,1] = arrow_len*np.sin(theta + self.dtheta[i])
        # clear current axis and replot
        self.ax.cla()
        self._config_axis()
        artist = self._draw_arrows()
        self._shading()
        return [artist,]
# %%
fig, ax = plt.subplots(1,1, figsize=(10,6),dpi=400)

dtheta = np.ones(80)*np.pi/160
dtheta = np.hstack((np.zeros(3), dtheta, np.zeros(5), -dtheta))
# create a figure updater
ud = UpdateFigure(ax, dtheta)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=dtheta.shape[0], blit=True)
# save animation as *.mp4
anim.save('dot_product_movie.mp4', fps=24, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%