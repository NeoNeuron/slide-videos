#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

#%%
class UpdateFigure:
    def __init__(self, ax:plt.Axes, data:np.ndarray):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): Axes of plot.
            data (np.ndarray): 2-D array of coordinate of each shot
        """

        self.colors = dict(
            init=[0,0,0,1],
            red=np.array([230,0,18,255])/255.0,
            green=np.array([0,176,80,255])/255.0,
        )
        self.ax = ax
        # generate a circle
        theta = np.linspace(0, 2*np.pi, 1000)
        self.ax.plot(np.cos(theta), np.sin(theta), color=self.colors['init'], lw=4, zorder=1)
        self.ax.plot(0.1*np.cos(theta), 0.1*np.sin(theta), color=self.colors['init'], lw=1, zorder=1)
        self.ax.set_xlim(-1.1,1.1)
        self.ax.set_ylim(-1.1,1.1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.data = data
        self.ax.axis('scaled')

        # initialize text
        self.n_chord = 0
        self.n_chord_hit = 0
        self.ax.set_xlabel(f'{0:5.3f}', fontsize=40)
        self.ax.set_title(f'{self.n_chord_hit:>5d}/{self.n_chord:<5d}', fontsize=40)

    
    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            if np.sum(self.data[i,:]**2) < 0.01:
                color='red'
                self.n_chord_hit += 1
            else:
                color='green'
            lines = self.ax.plot(self.data[i,0], self.data[i,1], 'x', color=self.colors[color], lw=1, zorder=10)
            self.n_chord += 1
            self.ax.set_title(f'{self.n_chord_hit:>5d}/{self.n_chord:<5d}', fontsize=40)
            self.ax.set_xlabel(f'{self.n_chord_hit*1.0/self.n_chord:5.3f}', fontsize=40)
        else:
            lines = self.ax.plot([], [])
        return lines

# ======================================================
# create canvas
fig, ax = plt.subplots(1,1, figsize=(5,5))
# Genearate random number
n_frame = 100
# ======================================================
# random seeds for area_chord
np.random.seed(202209071)
randx = np.random.rand(1000, 2)*2-1
mask = np.sum(randx**2, axis=1)<= 1
randx = randx[mask,:]
print(np.where(np.sum(randx**2, axis=1)<= 0.01))
# ======================================================
ud = UpdateFigure(ax, randx)
anim = FuncAnimation(fig, ud, frames=n_frame+1, blit=True)
# save animation as *.mp4
anim.save('missile_shot.mp4', fps=24, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# ======================================================
# %%