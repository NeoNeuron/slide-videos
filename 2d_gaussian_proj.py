# %%
from pathlib import Path
path = Path('./normal_2d/')
path.mkdir(exist_ok=True)
import numpy as np 
import matplotlib.pyplot as plt
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

plt.rcParams["font.size"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

#%%
#!================================================================================
# config data
from scipy.stats import multivariate_normal as mn
mean = np.zeros(2)
cov = np.eye(2)
xx, yy = np.meshgrid(np.linspace(-4,4,401), np.linspace(-4,4,401))
xysurf = mn.pdf(np.dstack((xx,yy)), mean, cov)
vmin, vmax = 0, np.max(xysurf)
# plot 2d heat map
fig, ax = plt.subplots(1,1,figsize=(6,6), 
                       gridspec_kw=dict(left=0.15, right=0.92, top=0.95, bottom=0.15, ))
ax.pcolormesh(xx,yy,xysurf, cmap='turbo',lw=0, alpha=0.9)
ax.axis('scaled')
ax.set_xlabel('x', fontsize=40)
ax.set_ylabel('y', fontsize=40)
ax.set_xticks([-4,-2,0,2,4])
ax.set_yticks([-4,-2,0,2,4])
fig.savefig(path/'normal3d_overview.png', dpi=300)
# plot 3d surface
fig = plt.figure(figsize=(6,5.5))
spec = plt.GridSpec(1, 1, 
    left=0.00, right=0.92, top=1, bottom=0.10, 
    figure=fig)
ax = fig.add_subplot(spec[0], projection='3d')
surf = ax.plot_surface(xx, yy, xysurf, cmap='turbo',
                rstride=1, cstride=1, vmin=vmin, vmax=vmax, alpha=0.9)

xticks=[-4,-2,0,2,4]
yticks=[-4,-2,0,2,4]
zticks=[0,0.1,.2]
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_zticks(zticks)
ax.set_xlim(xticks[0], xticks[-1])
ax.set_ylim(yticks[0], yticks[-1])
ax.set_zlim(zticks[0], zticks[-1])
ax.tick_params(axis='x', which='major', pad=-2)
ax.tick_params(axis='y', which='major', pad=-2)
ax.tick_params(axis='z', which='major', pad=5)
ax.set_xlabel('x', fontsize=40)
ax.set_ylabel('y', fontsize=40)

def lims(mplotlims):
    scale = 1.021
    offset = (mplotlims[1] - mplotlims[0])*scale
    return mplotlims[1] - offset, mplotlims[0] + offset
xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')

ax.view_init(10,)
plt.savefig(path/'normal3d.pdf')
plt.savefig(path/'normal3d.png', dpi=300, transparent=True)

ax.view_init(45, -55)
BLUE='#375492'
RED='#93391E'
proj_xz = xysurf.max(0)
proj_yz = xysurf.max(1)
ax.plot(xx[0,:], np.ones(yy.shape[0])*ylims[1], proj_xz, color=BLUE, lw=3)
ax.plot(np.ones(xx.shape[0])*xlims[0], yy[:,0], proj_yz, color=RED, lw=3)

stride=5
x_sparse = xx[0,::stride]
y_sparse = yy[::stride,0]
bar_width = (x_sparse[1]-x_sparse[0])/1.1
ax.bar3d(x_sparse-bar_width/2, np.ones_like(x_sparse)*ylims[1], 
    np.zeros_like(x_sparse), np.ones_like(x_sparse)*bar_width, np.zeros_like(x_sparse)*bar_width,
    proj_xz[::stride], color=BLUE, alpha=.5, shade=False,
    )
ax.bar3d(np.ones_like(y_sparse)*ylims[0], y_sparse-bar_width/2, 
    np.zeros_like(y_sparse), np.zeros_like(y_sparse)*bar_width, np.ones_like(y_sparse)*bar_width,
    proj_yz[::stride], color=RED, alpha=.5, shade=False,
    )

plt.savefig(path/'normal3d_proj.pdf')
plt.savefig(path/'normal3d_proj.png', dpi=300, transparent=True)
#%%