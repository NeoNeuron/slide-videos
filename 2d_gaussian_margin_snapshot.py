#%%
from pathlib import Path
path = Path('./normal_2d/')
path.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal as mn
# %%
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=14

COLORS = dict(
    blue        = '#375492',
    green       = '#88E685',
    dark_green  = '#00683B',
    red         = '#93391E',
    pink        = '#E374B7',
    purple      = '#A268B4',
    black       = '#000000',
)

class UpdateFigure:
    def __init__(self, ax1, ax2) -> None:

        # ====================
        # config data
        # ====================
        self.mean = np.array([0.48028984,0.5773781])
        self.cov = np.array([[0.03299816, 0.02951327],[0.02951327, 0.03009658]])
        self.y0=0.7
        self.cm = plt.cm.turbo
        self.xx, self.yy = np.meshgrid(np.linspace(0,1,201), np.linspace(0,1,201))
        xysurf = mn.pdf(np.dstack((self.xx,self.yy)), self.mean, self.cov)
        self.vmin, self.vmax = 0, np.max(xysurf)
        # ====================
        # plot 3d surface
        # ====================
        self.surf = ax1.plot_surface(self.xx, self.yy, xysurf, cmap=self.cm, alpha=0.4,
                        rstride=1, cstride=1, zorder=0, vmin=self.vmin, vmax=self.vmax)
        # ax1.view_init(10, None)
        ax1.set_xlabel(r'$x$(CPU负载)', fontsize=20, labelpad=10)
        ax1.set_ylabel(r'$y$(内存占用)', fontsize=20, labelpad=10)
        ax1.tick_params(axis='x', which='major', pad=2)
        ax1.tick_params(axis='y', which='major', pad=2)
        ax1.tick_params(axis='z', which='major', pad=4)
        ax1.xaxis.set_rotate_label(True)
        ax1.yaxis.set_rotate_label(True)
        # ax1.set_zlabel(r'$f(y)$', rotation=0, fontsize=20)
        xticks=[0,0.5,1]
        yticks=[0,0.5,1]
        zticks=[0,5,10,15]
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_zticks(zticks)
        ax1.set_xlim(xticks[0], xticks[-1])
        ax1.set_ylim(yticks[0], yticks[-1])
        ax1.set_zlim(zticks[0], zticks[-1])
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.xaxis.pane.set_edgecolor('w')
        ax1.yaxis.pane.set_edgecolor('w')
        ax1.zaxis.pane.set_edgecolor('w')
        snapshot()
        verts1, verts2 = self.get_verts(self.y0, self.mean, self.cov)
        self.shade1 = Poly3DCollection([verts1], 
            facecolor=COLORS['red'], edgecolor='None', alpha=1.0, zorder=1) # Add a polygon instead of fill_between
        ax1.add_collection3d(self.shade1)
        self.shade2 = Poly3DCollection([verts2], 
            facecolor=COLORS['green'], edgecolor='None', alpha=0.7, zorder=1) # Add a polygon instead of fill_between
        ax1.add_collection3d(self.shade2)
        snapshot()
        # ====================
        # draw conditional probability
        # ====================
        ax2.axis('on')
        p_cond = np.array(verts1[:-2])
        self.line, = ax2.plot(p_cond[:,0], p_cond[:,2], lw=5, color=COLORS['blue'])
        self.shade_2d = ax2.fill_between(p_cond[:,0],0, p_cond[:,2], color=COLORS['red'], alpha=0.8)
        ax2.set_xlabel(r'$x$(CPU负载)', fontsize=20)
        ax2.text(-0.155,0.5,r'$f(x|\qquad\qquad)$', fontsize=25,
            ha='center', va='center', rotation=90, color='k', transform=ax2.transAxes)
        ax2.set_ylabel(r'$y=%.2f$'%self.y0, color='red', fontsize=24, y=0.59)
        ax2.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
        ax2.tick_params(axis='x', which='major', pad=5)
        ax2.set_yticks([0,5,10,15])
        ax2.set_xlim(xticks[0], xticks[-1])
        ax2.set_ylim(zticks[0], zticks[-1])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        self.ax2 = ax2
        snapshot()
        # ====================
        # draw x1=0.3
        # ====================
        scatter = ax2.scatter([0.3],[0],s=200, marker='x', lw=4, c='#92D050',zorder=10, clip_on=False)
        print(mn.pdf((0.3,0.7), self.mean, self.cov))
        snapshot()
        scatter.remove()
        self.y0=0

    @staticmethod
    def get_verts(yp_, mean_, cov_):
        func = lambda x: mn.pdf((x,yp_), mean_, cov_)
        x_array = np.linspace(0,1,201)
        verts1 = [(xi, yp_, func(xi)) for xi in x_array] \
                + [(1, yp_, 0), (0, yp_, 0)]
        verts2 = [(xi, yp_, func(xi)) for xi in x_array] \
                + [(1, yp_, 15), (0, yp_, 15)]
        return verts1, verts2

    def __call__(self, i):
        if i==0:
            i = 90
        mean_ = self.mean
        cov_ = self.cov.copy()
        y_ = self.y0 + i/240
        # update 3d shades
        verts1, verts2 = self.get_verts(y_, mean_, cov_)
        self.shade1.set_verts([verts1])
        self.shade2.set_verts([verts2])
        # update 2d line
        p_cond = np.array(verts1[:-2])
        self.line.set_data(p_cond[:,0], p_cond[:,2])
        self.ax2.set_ylabel(r'$y=%.2f$'%y_, color='red', fontsize=22, y=0.59)
        # update shade
        verts_2d = [[v[0], v[2]] for v in verts1]
        self.shade_2d.set_verts([verts_2d])
        return [self.line,]

# ====================
# create canvas and config axis layouts
# ====================
fig = plt.figure(figsize=(10,4.5))
spec1 = gridspec.GridSpec(1, 1, 
    left=-0.05, right=0.50, top=1.05, bottom=0.08, 
    figure=fig)
spec2 = gridspec.GridSpec(1, 1, 
    left=0.6, right=0.96, top=0.88, bottom=0.24, 
    figure=fig)
ax1 = fig.add_subplot(spec1[0], projection='3d')
ax2 = fig.add_subplot(spec2[0])
ax2.axis('off')
counter = 0
def snapshot():
    global counter
    fig.savefig(path/f'2d_gaussian_margin_snapshot{counter:d}.pdf')
    fig.savefig(path/f'2d_gaussian_margin_snapshot{counter:d}.png', dpi=300)
    counter += 1

# create a figure updater
nframes=240
ud = UpdateFigure(ax1, ax2)
plt.savefig(path/'test_margin1.pdf')
#%%
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save(path/'cpu_ram_2d_margin.mp4', fps=24, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

# %%
