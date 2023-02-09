#%%
from pathlib import Path
path = Path('./covariance/')
path.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal as mn
from scipy.stats import norm
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
# %%

class UpdateFigure:
    def __init__(self, ax1, ax2, ax3, ax4, nframes, X, Y, rho, means, stds, xrange, yrange, mode):

        # ====================
        # config data
        # ====================
        self.mode = mode
        self.X = X
        self.Y = Y
        self.means = np.array(means)
        self.stds = np.array(stds)
        if self.mode == 0:
            self.dmeans = (0-self.means)/nframes
            self.dstds = (1-self.stds)/nframes
        else:
            self.dmeans = (0-self.means)/nframes*2
            self.dstds = (1-self.stds)/nframes*2
        self.nframes = nframes
        np.random.seed(1)

        self.xgrid = np.linspace(*xrange, 1000)
        self.axes = [ax1, ax2]
        self.axes_ylim_toggle = [False, False]
        self.axes_ylim_target = [None, None]
        self.axes_ylim_step = [None, None]
        for mean, std, ax, label in zip(means, stds, self.axes, ('X','Y') ):
            line, = ax.plot(self.xgrid, norm.pdf(self.xgrid, mean, std), color='k', lw=2)
            line.set_clip_on(False)
            self.lines.append(line)
            shade = ax.fill_between(self.xgrid, 0, norm.pdf(self.xgrid, mean, std), color='#1F77B4', alpha=0.5)
            self.shades.append(shade)
            ax.set_xlim(xrange)
            ax.set_ylim(0)
            ax.set_xlabel(label,fontsize=18)
            ax.set_ylabel(r'概率密度',fontsize=18)
        self.axes[0].set_ylim(0,0.4*1.05)

        self.scatter, = ax3.plot(X*stds[0]+means[0],Y*stds[1]+means[1],'o', 
                                 mec='k', mfc='#1F77B4', mew=0.5, alpha=1)
        ax3.set_xlim(xrange)
        ax3.set_ylim(yrange)
        ax3.set_xlabel(r'$X$',fontsize=18)
        ax3.set_ylabel(r'$Y$',fontsize=18)
        self.lines = []
        self.shades = []
        self.ax3 = ax3

        self.rho = rho
        cov = np.diag(self.stds**2)
        cov[0,1] = cov[1,0] = self.rho*self.stds[0]*self.stds[1]
        self.xx, self.yy = np.meshgrid(np.linspace(*xrange,201), np.linspace(*yrange,201))
        xysurf = mn.pdf(np.dstack((self.xx,self.yy)), self.means, cov)
        self.surf = ax4.pcolormesh(self.xx, self.yy, xysurf, cmap='turbo')
        ax4.set_xlabel(r'$X$',fontsize=18)
        ax4.set_ylabel(r'$Y$',fontsize=18)
        self.ax4 = ax4


    def __call__(self, i):

        if i>0 and i<=self.nframes:
            if self.mode == 0:
                # update means and stds
                means_new = self.means + self.dmeans*i
                stds_new = self.stds + self.dstds*i
                # update Gaussian curves and shades
                for idx, ax in enumerate(self.axes):
                    ydata = norm.pdf(self.xgrid, means_new[idx], stds_new[idx])
                    self.lines[idx].set_data(self.xgrid, ydata)
                    self.shades[idx].remove()
                    self.shades[idx] = ax.fill_between(
                        self.xgrid, 0, ydata, color='#1F77B4', alpha=0.5)
                # update scatter plot
                self.scatter.set_data(self.X*stds_new[0]+means_new[0],self.Y*stds_new[1]+means_new[1])
                self.surf.remove()
                # update heatmap for norm2d
                cov_new = np.diag(stds_new**2)
                cov_new[0,1] = cov_new[1,0] = self.rho*stds_new[0]*stds_new[1]
                self.surf = self.ax4.pcolormesh(
                    self.xx, self.yy, 
                    mn.pdf(np.dstack((self.xx,self.yy)), means_new, 
                        cov_new), cmap='turbo',
                    )
            elif self.mode == 1:
                # update means and stds
                if i <= self.nframes/2:
                    means_new = self.means + self.dmeans*i
                    stds_new = self.stds.copy()
                else:
                    means_new = self.means + self.dmeans*self.nframes/2
                    stds_new = self.stds + self.dstds*(i-self.nframes/2)
                # update Gaussian curves and shades
                for idx, ax in enumerate(self.axes[:1]):
                    ydata = norm.pdf(self.xgrid, means_new[idx], stds_new[idx])
                    self.lines[idx].set_data(self.xgrid, ydata)
                    self.shades[idx].remove()
                    self.shades[idx] = ax.fill_between(
                        self.xgrid, 0, ydata, color='#1F77B4', alpha=0.5)
            
        # update ylim
        for idx, ax in enumerate(self.axes):
            ymax = ax.get_ylim()[1]
            ydata = self.lines[idx].get_ydata()
            if not self.axes_ylim_toggle[idx]:
                if ydata.max() > ymax:
                    self.axes_ylim_toggle[idx] = True
                    self.axes_ylim_target[idx] = ymax*2
                    self.axes_ylim_step[idx] = (ymax*2-ymax)/20
                elif ydata.max() < ymax*0.31:
                    self.axes_ylim_toggle[idx] = True
                    self.axes_ylim_target[idx] = ymax/np.sqrt(10)
                    self.axes_ylim_step[idx] = (ymax/np.sqrt(10)-ymax)/20
            else:
                if np.abs(ymax - self.axes_ylim_target[idx])>1e-5:
                    ax.set_ylim(0, ymax+self.axes_ylim_step[idx])
                else:
                    self.axes_ylim_toggle[idx] = False

        return [self.scatter]

fig = plt.figure(figsize=(12,4))
spec = gridspec.GridSpec(2, 3, 
    left=0.08, right=0.98, top=0.95, bottom=0.15,
    hspace=0.5,
    wspace=0.2,
    figure=fig)
#fig.set_tight_layout(True)
ax1 = fig.add_subplot(spec[0,0], )
ax2 = fig.add_subplot(spec[1,0], )
ax3 = fig.add_subplot(spec[0:2,1])
ax4 = fig.add_subplot(spec[0:2,2])
# create a figure updater
nframes=144

BLUE = '#000089'
N = 1000
var = norm.rvs(size=(3,N))
rho = 0.5
X = var[0]*np.sqrt(1-rho)+var[2]*np.sqrt(rho)
Y = var[1]*np.sqrt(1-rho)+var[2]*np.sqrt(rho)

xrange = [-8,8]
yrange = [-8,8]
means = [2, 2]
stds  = [2, 0.1]

for mode in range(2):
    ud = UpdateFigure(ax1, ax2, ax3, ax4, nframes, X, Y, rho, means, stds, xrange, yrange, mode)
    # user FuncAnimation to generate frames of animation
    anim = FuncAnimation(fig, ud, frames=nframes+25, blit=True)
    # save animation as *.mp4
    anim.save(path/f'norm2d_morph_{mode}.mp4', fps=24, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%