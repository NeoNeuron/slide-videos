#%%
from init import *
from matplotlib.transforms import Affine2D
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

subway_marker = gen_marker(path.parents[1]/'icons/subway.svg',180)

### patch start https://stackoverflow.com/a/16496436
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
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
        ax.set_title(f"{self.hit:3d} / {0:3d}", fontsize=40)
        self.text=ax.text(-0.35,-0.2, 2.65, s=f'{0:5.2f}%', fontsize=40)
        self.ax = ax

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        nb = 100
        if i > 0 and i<=self.data.shape[0]/nb:
            if self.last_sample is not None:
                self.last_sample.set_alpha(0.2)
            # update lines
            ii = nb*(i-1)
            mask = np.sum(self.data[ii:ii+nb,:]**2, axis=1) < 1
            color = np.zeros((nb, 4))
            n_hit = mask.sum()
            color[mask,:]=np.tile(GREEN, (n_hit,1))
            color[~mask,:]=np.tile(RED, (nb-n_hit,1))
            self.current_sample = self.ax.scatter(self.data[ii:ii+nb,0],self.data[ii:ii+nb,1],self.data[ii:ii+nb,2], c=color, marker='x')
            self.hit += mask.sum()
            # self.current_sample, = self.ax.plot(*self.data[i-1,:], color=RED, marker='x')

            self.ax.set_title(f"{self.hit:3d} / {i*nb:3d}", fontsize=40)
            self.text.set_text(s=f'{self.hit*100.0/i/nb:5.2f}%')
            # update last sample
            self.last_sample = self.current_sample
        else:
            self.current_sample = self.ax.plot([],[],[])[0]
        return [self.current_sample,]
#%%
def gen_verts(cube_size):
    return [
        [-1, -1, 0], 
        [-1, cube_size[1]-1, 0], 
        [cube_size[0]-1, cube_size[1]-1, 0], 
        [cube_size[0]-1, -1, 0], 
        [-1, -1, cube_size[2]], 
        [-1, cube_size[1]-1, cube_size[2]], 
        [cube_size[0]-1, cube_size[1]-1, cube_size[2]], 
        [cube_size[0]-1, -1, cube_size[2]], 
        ]
fig = plt.figure(figsize=(10,10), dpi=100, )
gs = plt.GridSpec(1,1, top=0.95, bottom=0.05)
ax = fig.add_subplot(gs[0], projection='3d')

cube_size=np.ones(3)*2
verts = gen_verts(cube_size)
# face
faces = [
    [0, 1, 2, 3], 
    [0, 1, 5, 4], 
    [2, 3, 7, 6], 
    [0, 3, 7, 4], 
    [1, 2, 6, 5], 
    [4, 5, 6, 7],
    ]

# 
poly3d = [[verts[vert_id] for vert_id in face] for face in faces]
# draw vertex
# x, y, z = zip(*verts)
# self.ax.scatter(x, y, z)

collection = Poly3DCollection(poly3d, edgecolors= 'grey', facecolor= [0.5, 0.5, 0.5], linewidths=1, alpha=0.02)
ax.add_collection3d(collection)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(0,2)
# ax.set_xlabel('长', fontsize=40)
# ax.set_ylabel('宽', fontsize=40)
# ax.set_zlabel('高', fontsize=40)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.spines['right']=False
ax.view_init(15, -45)
ax.xaxis._axinfo["grid"]['linestyle'] = ":"
ax.yaxis._axinfo["grid"]['linestyle'] = ":"
ax.zaxis._axinfo["grid"]['linestyle'] = ":"

r = (1-np.linspace(0, 1, 20)**4)
theta = np.linspace(-1 * np.pi, 1 * np.pi, 120)
r, theta = np.meshgrid(r, theta)

xx = r * np.sin(theta)
yy = r * np.cos(theta)
zz = np.sqrt(1-xx**2-yy**2)
# ax.plot_surface(xx,yy,zz,color='grey',edgecolors='none',lw=0, alpha=0.5)
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.25, color='orange', lw=0)
plt.tight_layout()
plt.savefig(path/'test.pdf')
x = np.random.rand(10000,3)
x[:,0:2] = x[:,0:2]*2-1
x[:,2] = x[:,2]*2
# create a figure updater
ud = UpdateFigure_geo_prob(x, ax)
# %%
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=121, blit=True)
# save animation as *.mp4
anim.save(path/'geo_prob_3d.mp4', fps=24, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
