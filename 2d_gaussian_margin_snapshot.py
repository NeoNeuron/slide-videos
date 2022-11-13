#%%
from pathlib import Path
path = Path('./normal_2d/')
path.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal as mn
# %%
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
plt.rcParams['xtick.labelsize']=22
plt.rcParams['ytick.labelsize']=22

colors = dict(
    blue        = '#375492',
    green       = '#88E685',
    dark_green  = '#00683B',
    red         = '#93391E',
    pink        = '#E374B7',
    purple      = '#A268B4',
    black       = '#000000',
)

def get_verts(yp_, mean_, cov_):
    func = lambda x: mn.pdf((x,yp_), mean_, cov_)
    x_array = np.linspace(0,1,201)
    verts1 = [(xi, yp_, func(xi)) for xi in x_array] \
            + [(1, yp_, 0), (0, yp_, 0)]
    verts2 = [(xi, yp_, func(xi)) for xi in x_array] \
            + [(1, yp_, 15), (0, yp_, 15)]
    return verts1, verts2

# ====================
# config data
# ====================
mean = np.array([0.48028984,0.5773781])
cov = np.array([[0.03299816, 0.02951327],[0.02951327, 0.03009658]])
y0=0.7
cm = plt.cm.turbo
xx, yy = np.meshgrid(np.linspace(0,1,201), np.linspace(0,1,201))
xysurf = mn.pdf(np.dstack((xx,yy)), mean, cov)
vmin, vmax = 0, np.max(xysurf)

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
counter = 0
def snapshot():
    global counter
    fig.savefig(path/f'2d_gaussian_margin_snapshot{counter:d}.pdf')
    counter += 1

# ====================
# plot 3d surface
# ====================
ax1 = fig.add_subplot(spec1[0], projection='3d')
surf = ax1.plot_surface(xx, yy, xysurf, cmap=cm, alpha=0.4,
                rstride=1, cstride=1, zorder=0, vmin=vmin, vmax=vmax)
# ax1.view_init(10, None)
ax1.set_xlabel(r'$x$', fontsize=30, labelpad=20)
ax1.set_ylabel(r'$y$', fontsize=30, labelpad=20)
ax1.tick_params(axis='x', which='major', pad=8)
ax1.tick_params(axis='y', which='major', pad=8)
ax1.tick_params(axis='z', which='major', pad=5)
ax1.zaxis.set_rotate_label(False)
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
verts1, verts2 = get_verts(y0, mean, cov)
shade1 = Poly3DCollection([verts1], 
    facecolor=colors['red'], edgecolor='None', alpha=1.0, zorder=1) # Add a polygon instead of fill_between
ax1.add_collection3d(shade1)
shade2 = Poly3DCollection([verts2], 
    facecolor=colors['green'], edgecolor='None', alpha=0.7, zorder=1) # Add a polygon instead of fill_between
ax1.add_collection3d(shade2)
snapshot()

# ====================
# draw conditional probability
# ====================
ax2 = fig.add_subplot(spec2[0])
p_cond = np.array(verts1[:-2])
line, = ax2.plot(p_cond[:,0], p_cond[:,2], lw=5, color=colors['blue'])
shade_2d = ax2.fill_between(p_cond[:,0],0, p_cond[:,2], color=colors['red'], alpha=0.8)
ax2.set_xlabel(r'$x$', fontsize=30)
ax2.text(-0.185,0.5,r'$f(x|\qquad\qquad)$', fontsize=25,
    ha='center', va='center', rotation=90, color='k', transform=ax2.transAxes)
ax2.set_ylabel(r'$y=%.2f$'%y0, color='red', fontsize=24, y=0.59)
ax2.set_xticks([0,0.2,0.4,0.6,0.8,1.0], fontsize=20)
ax2.tick_params(axis='x', which='major', pad=10)
ax2.set_yticks([0,5,10,15], fontsize=20)
ax2.set_xlim(xticks[0], xticks[-1])
ax2.set_ylim(zticks[0], zticks[-1])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
snapshot()

# ====================
# draw x1=0.3 vertical line
# ====================
ax2.scatter([0.3],[0],s=600, marker='x', lw=8, c='#92D050',zorder=10).set_clip_on(False)
ax2.axvline(0.3, ls='--', lw=5, c='#92D050')
print(mn.pdf((0.3,0.7), mean, cov))
snapshot()