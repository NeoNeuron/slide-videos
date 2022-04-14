#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal as mn
# %%
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

class UpdateFigure:
    def __init__(self, ax1, ax2, ax3):

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
        self.xx, self.yy = np.meshgrid(np.linspace(-4,4,101), np.linspace(-4,4,101))
        xysurf = mn.pdf(np.dstack((self.xx,self.yy)), self.mean, self.cov)
        self.vmin, self.vmax = 0, np.max(xysurf)

        # ====================
        # draw LaTeX formula
        # ====================
        ax1.axis('off')
        self.tex = ax1.text(0.5,0.5,
            self.gen_text(self.mean, self.cov), 
            ha='center', va='top',
            color='k', fontsize=25)

        # ====================
        # plot 3d surface
        # ====================
        self.surf = ax2.plot_surface(self.xx, self.yy, xysurf, cmap=self.cm,
                        rstride=1, cstride=1, vmin=self.vmin, vmax=self.vmax)
        ax2.view_init(10, None)

        ax2.set_xlabel(r'$x$', fontsize=20)
        ax2.set_ylabel(r'$y$', fontsize=20)
        ax2.zaxis.set_rotate_label(False)
        # ax2.set_zlabel(r'$f(y)$', rotation=0, fontsize=20)
        xticks=[-4,-2,0,2,4]
        yticks=[-4,-2,0,2,4]
        zticks=[0,.1,.2]
        ax2.set_xticks(xticks)
        ax2.set_yticks(yticks)
        ax2.set_zticks(zticks)
        ax2.set_xlim(xticks[0], xticks[-1])
        ax2.set_ylim(yticks[0], yticks[-1])
        ax2.set_zlim(zticks[0], zticks[-1])
        # ax2.invert_xaxis()
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor('w')
        ax2.yaxis.pane.set_edgecolor('w')
        ax2.zaxis.pane.set_edgecolor('w')
        self.ax2 = ax2

        # ====================
        # draw 2d pcolor
        # ====================
        self.mesh = ax3.pcolormesh(self.xx, self.yy, xysurf, cmap=self.cm)
        ax3.axis('scaled')

    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\mu}=\begin{bmatrix}%.1f\\%.1f\end{bmatrix},\boldsymbol{\Sigma}=\begin{bmatrix}%.1f & %.1f \\ %.1f & %.1f\end{bmatrix}$"%(*_mean, *_cov.flatten())

    def __call__(self, i):
        mean_ = self.mean
        cov_ = self.cov.copy()
        cov_[1,1] += i*0.02
        xysurf = mn.pdf(np.dstack((self.xx,self.yy)), mean_, cov_)

        self.surf.remove()
        self.surf = self.ax2.plot_surface(self.xx, self.yy, xysurf, cmap=self.cm,
                        rstride=1, cstride=1, vmin=self.vmin, vmax=self.vmax)
        self.mesh.set_array(xysurf)
        self.tex.set_text(self.gen_text(mean_, cov_), )
        return [self.surf,]

fig = plt.figure(figsize=(5,10),dpi=400,)
spec = gridspec.GridSpec(3, 1, 
    left=0.10, right=0.90, top=1, bottom=0.05, 
    height_ratios=[1,5,4],
    figure=fig)
ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1], projection='3d')
ax3 = fig.add_subplot(spec[2], )
# create a figure updater
nframes=100
ud = UpdateFigure(ax1, ax2, ax3)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save('2d_gaussian.mp4', fps=20, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
