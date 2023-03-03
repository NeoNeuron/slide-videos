#%%
from pathlib import Path
path = Path('./geometric_probability/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
from matplotlib.animation import FuncAnimation
def gen_marker(fname:str, rotation:float=180):
    """Generate maker from svg image file.

    Args:
        fname (str): filename of svg image.
        rotation (int, optional): 
            degree of rotation of original images. Defaults to 180.

    Returns:
        Object of marker.
    """
    person_path, attributes = svg2paths(fname)
    person_marker = parse_path(attributes[0]['d'])
    person_marker.vertices -= person_marker.vertices.mean(axis=0)
    person_marker = person_marker.transformed(Affine2D().rotate_deg(rotation))
    person_marker = person_marker.transformed(Affine2D().scale(-1,1))
    return person_marker

subway_marker = gen_marker('icons/subway.svg',180)
#%%
RED=np.array([230,0,18,255])/255.0
GREEN=np.array([0,176,80,255])/255.0
class UpdateFigure_geo_prob:
    def __init__(self, data:np.ndarray, ax:plt.Axes):
        """Plot the first frame for the animation.

        Args:
            data (np.ndarray): 1-D array of number of passagers for each days
            ax (plt.Axes): axes of scatter plot
        """

        self.data = data
        self.trials = np.arange(data.shape[0])+1

        # vertical lines:
        self.last_sample = None
        self.current_sample = None
        self.hit = 0
        ax.set_title(f"{self.hit:3d} / {0:3d}", fontsize=20, pad=10)
        self.ax = ax

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0 and i<=self.data.shape[0]:
            if self.last_sample is not None:
                self.last_sample.set_alpha(0.5)
            # update lines
            if self.data[i-1] < 0.5:
                self.current_sample = self.ax.axvline(self.data[i-1], color=RED, lw=0.8)
                self.hit += 1
            else:
                self.current_sample = self.ax.axvline(self.data[i-1], color=GREEN, lw=0.8)

            self.ax.set_title(f"{self.hit:3d} / {i:3d}", fontsize=20, pad=10)
            # update last sample
            self.last_sample = self.current_sample
        else:
            self.current_sample = self.ax.plot([],[])[0]
        return [self.current_sample,]
#%%
fig, ax = plt.subplots(
    1,1, figsize=(10,2), dpi=100, 
    gridspec_kw={'top':0.6, 'bottom':0.45, 'left':0.15, 'right':0.85})
ax.plot([-0.5,],[0,], ms=100, color='k', marker=subway_marker, clip_on=False)
ax.plot([5.5,],[0,], ms=100, color='k', marker=subway_marker, clip_on=False)
ax.fill_betweenx([-1,1], 0, 0.5, color='grey', alpha=0.2)
ax.set_xlim(0,5)
ax.set_ylim(-1,1)
ax.set_yticks([])
ax.set_xticks(np.arange(6))
ax.set_xticklabels(ax.get_xticks(), fontsize=20)
ax.set_xlabel('乘客到站时间(分钟)', fontsize=25)
np.random.seed(5)
x = np.random.rand(500)*5
# create a figure updater
ud = UpdateFigure_geo_prob(x, ax)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=576, blit=True)
# save animation as *.mp4
anim.save(path/'geo_prob_1d.mp4', fps=96, dpi=300, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
fig.savefig(path/'geo_prob_1d_finalshot.png', dpi=300)
# %%
