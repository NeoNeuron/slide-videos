#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

@np.vectorize
def circ(x):
    if x > 2*np.pi:
        return x-2*np.pi
    elif x < 0:
        return x+2*np.pi
    else:
        return x

def get_phi(x,y):
    phi = np.arctan(y/x)
    if x < 0:
        phi = np.pi + phi
    return phi
#%%
class UpdateFigure:
    def __init__(self, ax:plt.Axes, data:np.ndarray):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
            data (np.ndarray): 1-D array of number of passagers for each days
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
        # array to record the color of each flight
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

    @staticmethod
    def vert_chord(seed,):
        xs = np.ones(2)*(seed*2-1)
        ys = np.sqrt(1 - xs**2)
        ys[1] *= -1
        return xs, ys

    @staticmethod
    def radiate_chord(seed,):
        xs, ys = np.zeros(2), np.zeros(2)
        xs[0], ys[0] = 1, 0
        xs[1], ys[1] = np.cos(seed*2*np.pi), np.sin(seed*2*np.pi)
        return xs, ys

    @staticmethod
    def area_chord(seed,):
        xc, yc = seed*2-1
        rho = np.sqrt((seed[0]*2-1)**2+(seed[1]*2-1)**2)
        phi = get_phi(xc,yc)
        angle = np.arccos(rho)
        theta = circ(np.array([phi-angle, phi+angle]))
        xs, ys = np.cos(theta), np.sin(theta)
        return xs, ys

    @staticmethod
    def chord_len(xs, ys):
        return np.sqrt((xs[0]-xs[1])**2+(ys[0]-ys[1])**2)
    
    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            xs, ys = self.area_chord(self.data[i]) # can be replaced by vert_chord and radiate_chord
            if self.chord_len(xs, ys) > np.sqrt(3):
                lines = self.ax.plot(xs, ys, color=self.colors['red'], lw=1, zorder=0)
                self.n_chord_hit += 1
            else:
                lines = self.ax.plot(xs, ys, color=self.colors['green'], lw=1, zorder=0)
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
n_frame = 500
# ======================================================
# random seeds for vert_chord and radiate_chord
# randx = np.random.rand(n_frame+1)
# ======================================================
# random seeds for area_chord
randx = np.random.rand(n_frame*2, 2)
mask = (randx[:,0]-0.5)**2 + (randx[:,1]-0.5)**2<= 0.25
randx = randx[mask,:]
# ======================================================
ud = UpdateFigure(ax, randx)
anim = FuncAnimation(fig, ud, frames=n_frame+1, blit=True)
# save animation as *.mp4
anim.save('rcp_movie.mp4', fps=24, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# ======================================================
# %%