#%%
from pathlib import Path
path = Path('./normal_2d/')
path.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal as mn
# %%
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15

class UpdateFigure:
    def __init__(self, ax1, ax2,):

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
        self.mean = np.zeros(2)
        self.cov = np.eye(2)
        self.xx, self.yy = np.meshgrid(np.linspace(-4,4,401), np.linspace(-4,4,401))
        xysurf = mn.pdf(np.dstack((self.xx,self.yy)), self.mean, self.cov)*2
        xysurf[self.xx*self.yy>0] = 0
        self.vmin, self.vmax = 0, np.max(xysurf)

        # ====================
        # plot 3d surface
        # ====================
        self.surf = ax1.plot_surface(self.xx, self.yy, xysurf, cmap=self.cm,
                        rstride=1, cstride=1, vmin=self.vmin, vmax=self.vmax)
        ax1.view_init(10, None)

        ax1.set_xlabel(r'$x$', fontsize=20, labelpad=8)
        ax1.set_ylabel(r'$y$', fontsize=20, labelpad=8)
        ax1.tick_params(axis='x', which='major', pad=-2)
        ax1.tick_params(axis='y', which='major', pad=-2)
        ax1.tick_params(axis='z', which='major', pad=5)
        ax1.zaxis.set_rotate_label(False)
        # ax1.set_zlabel(r'$f(y)$', rotation=0, fontsize=20)
        xticks=[-4,-2,0,2,4]
        yticks=[-4,-2,0,2,4]
        zticks=[0,.2,.4]
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_zticks(zticks)
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylim(yticks[0], yticks[-1])
        ax1.set_zlim(zticks[0], zticks[-1])
        # ax1.invert_xaxis()
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.xaxis.pane.set_edgecolor('w')
        ax1.yaxis.pane.set_edgecolor('w')
        ax1.zaxis.pane.set_edgecolor('w')
        self.ax1 = ax1

        proj_xz = xysurf.max(0)
        proj_yz = xysurf.max(1)
        xlims=[-4,4]
        ylims=[-4,4]
        ax1.plot(self.xx[0,:], np.ones(self.yy.shape[0])*ylims[1], proj_xz, color='#E0A419', lw=2.5)
        ax1.plot(np.ones(self.xx.shape[0])*xlims[0], self.yy[:,0], proj_yz, color='#88AD80', lw=2.5)

        # ====================
        # draw 2d pcolor
        # ====================
        self.mesh = ax2.pcolormesh(self.xx, self.yy, xysurf, cmap=self.cm, lw=0)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.axis('scaled')
        ax2.set_xticks([-4,-2,0,2,4])
        ax2.set_yticks([-4,-2,0,2,4])
        ax2.set_xlabel(r'$x$', fontsize=20, labelpad=0)
        ax2.set_ylabel(r'$y$', fontsize=20, rotation=0, va='center', labelpad=5)

        self.dangle = 1

    def __call__(self, i):
        self.ax1.view_init(None, -60+i*self.dangle)
        return [self.surf,]

fig = plt.figure(figsize=(6,3))
spec = gridspec.GridSpec(1, 2, 
    left=0.00, right=0.92, top=1, bottom=0.10, 
    wspace=0.1,
    width_ratios=[6,4], figure=fig)
ax1 = fig.add_subplot(spec[0], projection='3d')
ax2 = fig.add_subplot(spec[1])

# create a figure updater
nframes=360
ud = UpdateFigure(ax1, ax2)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save(path/'2d_gaussian_rotate.mp4', fps=24, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
