# %%
import numpy as np
import matplotlib as mpl
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#%%
class UpdateFigure:
    def __init__(self, 
        ax:plt.Axes,
        frames:int=10):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
        """

        self.ax = ax
        # plot tetrahedron
        verts = [
            [np.sqrt(8/9), 0, -1/3], 
            [-np.sqrt(2/9), np.sqrt(2/3), -1/3], 
            [-np.sqrt(2/9), -np.sqrt(2/3), -1/3], 
            [0, 0, 1], 
            [0, 0, -1/3], 
            ]
        # face
        faces = [
            # [0, 1, 2], 
            [0, 1, 3], 
            [0, 2, 3], 
            [1, 2, 3],
            [4, 1, 2],
            [0, 4, 2],
            [0, 1, 4],
            ]
        x, y, z = zip(*verts)
        self.ax.scatter(x, y, z, c='grey')

        poly3d = [[verts[vert_id] for vert_id in face] for face in faces]
        self.collection = Poly3DCollection(poly3d, edgecolors= 'grey', facecolor= ['r', '#FFFB00', 'b', '#FFFB00', 'r', 'b'], linewidths=1, alpha=1)
        self.ax.add_collection3d(self.collection)
        self.ax.set_xlim(-1,1)
        self.ax.set_ylim(-1,1)
        self.ax.set_zlim(-1,1)
        # self.ax.set_xlabel('X')
        # self.ax.set_ylabel('Y')
        # self.ax.set_zlabel('Z')
        # self.ax.set_xticklabels([])
        # self.ax.set_yticklabels([])
        # self.ax.set_zticklabels([])
        self.ax.axis('off')
        self.ax.xaxis._axinfo["grid"]['linestyle'] = ":"
        self.ax.yaxis._axinfo["grid"]['linestyle'] = ":"
        self.ax.zaxis._axinfo["grid"]['linestyle'] = ":"

        self.dangle = 360/nframes*1.4
        self.nframes = nframes
        self.split_frame = int(self.nframes/4*3)
        self.line =self.ax.plot([],[])

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i < self.split_frame:
            self.ax.view_init(15, 100+i*self.dangle)
        else:
            self.ax.view_init(15-(i-self.split_frame)*self.dangle, 100+self.split_frame*self.dangle)
        return self.line
# %%
fig = plt.figure(figsize=(6,6),dpi=200,)
ax = fig.add_subplot(projection='3d')

# create a figure updater
nframes=360
ud = UpdateFigure(ax, nframes)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save('tetrahedron_movie.mp4', fps=60, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%