# %%
from pathlib import Path
path = Path('./LinearRegression/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d

        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

plt.rcParams['font.size'] = 16
# %%
class UpdateFigure:
    def __init__(self, ax, arrow, b, b_parallel, line, traj):

        self.colors = dict(
            blue        = '#375492',
            green       = '#88E685',
            dark_green  = '#00683B',
            red         = '#93391E',
            pink        = '#E374B7',
            purple      = '#A268B4',
            black       = '#000000',
        )
        self.cm = plt.cm.turbo
        # ====================
        # config data
        # ====================
        self.traj = traj
        self.ax = ax
        self.arrow = arrow
        self.line = line
        self.b = b
        self.b_parallel = b_parallel
        self.dot = None

    def __call__(self, i):

        if i < self.traj.shape[1]:
            xj, yj, zj = self.traj
            # update arrow
            self.arrow.remove()
            self.arrow = Arrow3D([0,xj[i]],[0,yj[i]],[0,zj[i]],
                                mutation_scale=10, zorder=10, shrinkB=0,
                                lw=2, arrowstyle="->", color="darkgreen")
            self.ax.add_artist(self.arrow)
            # update 2d line
            ratio=1
            self.line.remove()
            self.line=Arrow3D(
                    [self.b[0]*ratio,xj[i]*ratio],
                    [self.b[1]*ratio,yj[i]*ratio],
                    [self.b[2]*ratio,zj[i]*ratio], 
                    mutation_scale=10,
                    zorder=10, shrinkB=0, lw=2, arrowstyle="<-", color='r')
            self.ax.add_artist(self.line)
            # update title
            length = np.sqrt(np.sum((np.array([xj[i], yj[i], zj[i]])-self.b)**2))
            self.ax.set_title(f'长度={length:.3f}', color='#8D0000')
        elif i >= self.traj.shape[1]+12:
            if self.dot is None:
                self.dot, = self.ax.plot(self.traj[0,0],self.traj[1,0],self.traj[2,0], 'o', ms=4, zorder=11, color='yellow')
                # self.dot, = self.ax.plot(self.traj[0,-1],self.traj[1,-1],self.traj[2,-1], 'o', ms=4, zorder=11, color='yellow')
        return [self.line,]
#%%
# ====================
# create canvas and config axis layouts
# ====================
plane_vec = np.array([0,0,1], dtype=float)
plane_vec /= np.sqrt(np.sum(plane_vec**2))
x = np.arange(-16,16)
y = np.arange(-16,16)
xx, yy = np.meshgrid(y, y)
zz = (-xx*plane_vec[0] - yy*plane_vec[1])/plane_vec[2]


fig = plt.figure(figsize=(5,4),)
spec = plt.GridSpec(1, 1, 
    left=-0.1, right=1.1, top=1, bottom=-0.10,
    figure=fig)
ax = fig.add_subplot(spec[0], projection='3d')
# config figure view
ax.view_init(20, 70)
ax.set_xlim(-15,15)
ax.set_ylim(-15,15)
ax.set_zlim(0,8)
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
# ax.set_xlabel(r'$\beta_0$', labelpad=-10)
# ax.set_ylabel(r'$\beta_1$', labelpad=-10)
ax.set_zlabel(r'')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.set_box_aspect(aspect = (1,1,0.5))

counter = 0
def snapshot():
    global counter
    fig.savefig(path/f'lsqr_3d_snapshot{counter:d}.png', dpi=300)
    counter += 1
# plot origin
ax.plot(0,0,0, 'o', zorder=11, color='navy', ms=3)
snapshot()
b = np.array([6.423,-4.157,8.713], dtype=float)
b_parallel = b-(plane_vec*b).sum()*plane_vec
a = Arrow3D([0,b[0]],[0,b[1]],[0,b[2]],
            mutation_scale=10, zorder=10,
            lw=2, arrowstyle="->", color="orange")
ax.add_artist(a)
snapshot()
# plot x0 and x1
a = Arrow3D([0,b_parallel[0]],[0,0],[0,0],
            mutation_scale=10, zorder=10, alpha=0.8,
            lw=2, arrowstyle="->", color="gray")
ax.add_artist(a)
snapshot()
a = Arrow3D([0,0],[0,b_parallel[1]*1.4],[0,0],
            mutation_scale=10, zorder=10, alpha=0.8,
            lw=2, arrowstyle="->", color="gray")
ax.add_artist(a)
snapshot()
# plot X*beta
a = Arrow3D([0,b_parallel[0]],[0,b_parallel[1]],[0,b_parallel[2]],
            mutation_scale=10, zorder=10, shrinkA=0,
            lw=2, arrowstyle="->", color="darkgreen")
ax.add_artist(a)
snapshot()
# plot plane containing X*beta
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.3, color='darkgreen', ec='darkgreen',
                lw=0, antialiased=False, shade=False, zorder=0)
snapshot()
# plot epsilon
ratio=1
line=Arrow3D(
        [b[0]*ratio,b_parallel[0]*ratio],
        [b[1]*ratio,b_parallel[1]*ratio],
        [b[2]*ratio,b_parallel[2]*ratio], 
        mutation_scale=10,
        zorder=10, shrinkB=0, lw=2, arrowstyle="<-", color='r')
ax.add_artist(line)
snapshot()

length = np.sqrt(np.sum((b_parallel-b)**2))
ax.set_title(f'长度={length:.3f}', color='#8D0000', y=0.93)

from scipy.interpolate import interp1d
theta = np.arange(0,1.01,0.1)*np.pi*2 -np.pi
theta_dense = np.arange(0,1.001,0.005)*np.pi*2 -np.pi
np.random.seed(2022)
rho = 5+np.random.randn(theta.shape[0])*2
rho = np.append(rho, np.sqrt(b_parallel[0]**2+b_parallel[1]**2))
theta_star = np.arctan(b_parallel[1]/b_parallel[0])
# if theta_star<0:
#     theta_star += np.pi*2
theta = np.append(theta, theta_star)
sortid=np.argsort(theta)
theta = theta[sortid]
rho = rho[sortid]
rho[0] = rho[-1]
rho_iterp= interp1d(theta, rho, kind='cubic')
x_trajectory = rho_iterp(theta_dense)*np.cos(theta_dense)
y_trajectory = rho_iterp(theta_dense)*np.sin(theta_dense)
z_trajectory = (-x_trajectory*plane_vec[0] - y_trajectory*plane_vec[1])/plane_vec[2]
# ax.plot(x_trajectory,y_trajectory,z_trajectory, zorder=11, color='yellow', lw=1)
idx = 82
# ax.plot(x_trajectory[idx],y_trajectory[idx],z_trajectory[idx], 'o', ms=4, zorder=11, color='orange')
# ax.plot(b_parallel[0],b_parallel[1],b_parallel[2], 'o', ms=4, zorder=11, color='yellow')

x_trajectory = np.hstack((x_trajectory[idx:], x_trajectory[:idx+1]))
y_trajectory = np.hstack((y_trajectory[idx:], y_trajectory[:idx+1]))
z_trajectory = np.hstack((z_trajectory[idx:], z_trajectory[:idx+1]))
traj = np.array([x_trajectory, y_trajectory, z_trajectory])

plt.savefig(path/'lsq-3d-v2.png', dpi=300)
#%%
# create a figure updater
nframes=240
ud = UpdateFigure(ax, a, b, b_parallel, line, traj)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save(path/'lsq-3d-v2.mp4', fps=24, dpi=300, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
#%%