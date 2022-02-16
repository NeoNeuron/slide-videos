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
        ax:plt.Axes, cube_size:np.ndarray=np.ones(3), 
        cube_target_size:np.ndarray=np.ones(3), 
        frames:int=10):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
        """

        self.ax = ax
        # plot cube
        # 顶点坐标
        verts = self.gen_verts(cube_size)
        # face
        self.faces = [
            [0, 1, 2, 3], 
            [0, 1, 5, 4], 
            [2, 3, 7, 6], 
            [0, 3, 7, 4], 
            [1, 2, 6, 5], 
            [4, 5, 6, 7],
            ]
        
        # 
        poly3d = [[verts[vert_id] for vert_id in face] for face in self.faces]
        # draw vertex
        # x, y, z = zip(*verts)
        # self.ax.scatter(x, y, z)
        
        self.collection = Poly3DCollection(poly3d, edgecolors= 'r', facecolor= [0.5, 0.5, 1], linewidths=1, alpha=0.3)
        self.ax.add_collection3d(self.collection)
        self.ax.set_xlim(0,4)
        self.ax.set_ylim(0,4)
        self.ax.set_zlim(0,4)
        self.ax.set_xlabel('长')
        self.ax.set_ylabel('宽')
        self.ax.set_zlabel('高')
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        self.init_size=cube_size
        self.ax.xaxis._axinfo["grid"]['linestyle'] = ":"
        self.ax.yaxis._axinfo["grid"]['linestyle'] = ":"
        self.ax.zaxis._axinfo["grid"]['linestyle'] = ":"

        self.dsize = (cube_target_size-cube_size)/frames
        self.line =self.ax.plot([],[])

    @staticmethod
    def gen_verts(cube_size):
        return [
            [0, 0, 0], 
            [0, cube_size[1], 0], 
            [cube_size[0], cube_size[1], 0], 
            [cube_size[0], 0, 0], 
            [0, 0, cube_size[2]], 
            [0, cube_size[1], cube_size[2]], 
            [cube_size[0], cube_size[1], cube_size[2]], 
            [cube_size[0], 0, cube_size[2]], 
            ]

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        verts = self.gen_verts(self.init_size+i*self.dsize)
        poly3d = [[verts[vert_id] for vert_id in face] for face in self.faces]# 画顶点
        # x, y, z = zip(*verts)
        # self.ax.scatter(x, y, z)
        self.collection.set_verts(poly3d)
        return self.line
# %%
fig = plt.figure(figsize=(6,6),dpi=200,)
ax = fig.add_subplot(projection='3d')
# create a figure updater
nframes=60
ud = UpdateFigure(ax, np.ones(3), np.array([2,3,1./6]) ,nframes)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save('cube_movie.mp4', fps=60, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
fig = plt.figure(figsize=(5,5),dpi=400,)
ax = fig.add_subplot(projection='3d')
nframes=60
ud = UpdateFigure(ax, np.ones(3), np.array([2,3,1./6]) ,nframes)
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
anim.save('cube_movie1.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

fig = plt.figure(figsize=(5,5),dpi=400,)
ax = fig.add_subplot(projection='3d')
nframes=60
ud = UpdateFigure(ax, np.array([2,3,1./6]), np.array([4,0.5,0.5]) ,nframes)
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
anim.save('cube_movie2.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

fig = plt.figure(figsize=(5,5),dpi=400,)
ax = fig.add_subplot(projection='3d')
nframes=60
ud = UpdateFigure(ax, np.array([4,0.5,0.5]), np.array([2./3,1,1.5]) ,nframes)
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
anim.save('cube_movie3.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
