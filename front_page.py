# %%
import numpy as np 
import matplotlib.pyplot as plt
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

#%%
data = np.loadtxt('data.csv', delimiter='\t')
_x = np.arange(5)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
#%%
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.bar3d(_xx.flatten(), _yy.flatten(), 
    np.zeros(25), np.ones(25), np.ones(25),
    data.flatten()*10, color=plt.cm.YlGnBu_r(data.flatten()/data.max())
    )
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.view_init(15, -45)
ax.set_zticks([0, 0.5, 1])
ax.set_zticklabels([])
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.savefig('fp_hist3d.pdf', transparent=True)
# %%
from scipy.stats import norm
xdata = np.linspace(-4, 4, 201)
fig, ax = plt.subplots(1,1, figsize=(7,5), )
for sigma in (0.5, 1, 2):
    ax.plot(xdata, norm.pdf(xdata, scale=sigma), lw=4, color=plt.cm.YlGnBu(sigma/2))
ax.set_xlim(-4,4)
ax.set_ylim(0, 0.8)
ax.set_xticks(np.arange(-4,4.1,2))
ax.set_yticks(np.arange(0,0.8,0.1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['left'].set_position('center')
plt.savefig('fp_normal3.pdf', transparent=True)
# %%

xdata = np.linspace(-4, 4, 201)
xdata_sparse = np.linspace(-3.9,3.9,19)
fig, ax = plt.subplots(1,1, figsize=(7,5), )
ax.plot(xdata, norm.pdf(xdata), lw=4, color=plt.cm.YlGnBu(2.5/3))
ax.bar(xdata_sparse, norm.pdf(xdata_sparse), 
    width = (xdata_sparse[1]-xdata_sparse[0])/2,lw=2, 
    ec='grey', fc=plt.cm.YlGnBu(1./2))
ax.set_xlim(-4,4)
ax.set_ylim(0, 0.45)
ax.set_xticks(np.arange(-4,4.1,2))
ax.set_yticks(np.arange(0,0.5,0.1))
ax.set_xticklabels([])
ax.set_yticks([])
ax.spines['left'].set_visible(False)
plt.savefig('fp_normal_hist.pdf', transparent=True)

#%%
#!================================================================================
# config data
from scipy.stats import multivariate_normal as mn
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
mean = np.zeros(2)
cov = np.eye(2)
xx, yy = np.meshgrid(np.linspace(-4,4,101), np.linspace(-4,4,101))
xysurf = mn.pdf(np.dstack((xx,yy)), mean, cov)
vmin, vmax = 0, np.max(xysurf)
# plot 3d surface
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(xx, yy, xysurf, cmap='YlGnBu_r',
                rstride=1, cstride=1, vmin=vmin, vmax=vmax)
ax.view_init(20, 40)

xticks=[-4,-2,0,2,4]
yticks=[-4,-2,0,2,4]
zticks=[0,0.2/3,0.4/3,.2]
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlim(xticks[0], xticks[-1])
ax.set_ylim(yticks[0], yticks[-1])
ax.set_zlim(zticks[0], zticks[-1])

def lims(mplotlims):
    scale = 1.021
    offset = (mplotlims[1] - mplotlims[0])*scale
    return mplotlims[1] - offset, mplotlims[0] + offset
xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())

verts = [
    [xlims[0], ylims[0], zlims[0]],
    [xlims[0], ylims[1], zlims[0]],
    [xlims[1], ylims[1], zlims[0]],
    [xlims[1], ylims[0], zlims[0]],
    [xlims[0], ylims[0], zlims[1]],
    [xlims[0], ylims[1], zlims[1]],
    [xlims[1], ylims[1], zlims[1]],
    [xlims[1], ylims[0], zlims[1]],
    ]
# face
faces = [
    # [0, 1, 2, 3], 
    # [0, 1, 5, 4], 
    # [2, 3, 7, 6], 
    # [0, 3, 7, 4], 
    # [1, 2, 6, 5], 
    [4, 5, 6, 7],
    ]
        
# poly3d = [[verts[vert_id] for vert_id in face] for face in faces]

# p = Poly3DCollection(poly3d, fc=[0,0,0,0], ec='grey', lw=1)
# ax.add_collection3d(p)

# p = Poly3DCollection([[verts[0],verts[4]]], fc=[0,0,0,0], ec='grey', lw=1)
# ax.add_collection3d(p)
# p = Poly3DCollection([[verts[1],verts[5]]], fc=[0,0,0,0], ec='grey', lw=1)
# ax.add_collection3d(p)
# p = Poly3DCollection([[verts[2],verts[6]]], fc=[0,0,0,0], ec='grey', lw=1)
# ax.add_collection3d(p)


# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# ax.xaxis.pane.set_edgecolor('w')
# ax.yaxis.pane.set_edgecolor('w')
# ax.zaxis.pane.set_edgecolor('w')

plt.savefig('fp_normal3d.pdf', transparent=True)

proj_xz = xysurf.max(0)
proj_yz = xysurf.max(1)
ax.plot(xx[0,:], np.ones(yy.shape[0])*ylims[0], proj_xz, color='#E0A419', lw=3)
ax.plot(np.ones(xx.shape[0])*xlims[0], yy[:,0], proj_yz, color='#88AD80', lw=3)

stride=5
x_sparse = xx[0,::stride]
y_sparse = yy[::stride,0]
bar_width = (x_sparse[1]-x_sparse[0])/2
ax.bar3d(x_sparse-bar_width/2, np.ones_like(x_sparse)*ylims[0], 
    np.zeros_like(x_sparse), np.ones_like(x_sparse)*bar_width, np.zeros_like(x_sparse)*bar_width,
    proj_xz[::stride], color='#E0A419', alpha=.5, shade=False,
    )
ax.bar3d(np.ones_like(y_sparse)*ylims[0], y_sparse-bar_width/2, 
    np.zeros_like(y_sparse), np.zeros_like(y_sparse)*bar_width, np.ones_like(y_sparse)*bar_width,
    proj_yz[::stride], color='#88AD80', alpha=.5, shade=False,
    )

plt.savefig('fp_normal3d_proj.pdf', transparent=True)

# %%
#!================================================================================
mean = np.zeros(2)
cov = np.array([[2.0,0],[0,0.5]])
xx, yy = np.meshgrid(np.linspace(-4,4,101), np.linspace(-4,4,101))
xysurf = mn.pdf(np.dstack((xx,yy)), mean, cov)
fig,ax = plt.subplots(1,1,figsize=(6,6))
mesh = ax.pcolormesh(xx, yy, xysurf, cmap='YlGnBu_r', lw=0, rasterized=True)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.axis('scaled')
plt.savefig('fp_normal2d.pdf', transparent=True)
# %%
#!================================================================================
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')

# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(0, 1.25, 100)
p = np.linspace(0, 2*np.pi, 50)
R, P = np.meshgrid(r, p)
# Z = ((R**2 - 1)**2)
Z = 0.5*(np.cos(R*2*np.pi)+1)*np.exp(-R**2)

# Express the mesh in the cartesian system.
X, Y = R*np.cos(P), R*np.sin(P)

# Plot the surface.
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r)

ax.set_box_aspect((1,1,0.5))
ax.view_init(20, 40)
# Tweak the limits and add latex math labels.
ax.set_zlim(0, 1)
ax.set_zticks([0,0.5,1])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# ax.set_xlabel(r'$\phi_\mathrm{real}$')
# ax.set_ylabel(r'$\phi_\mathrm{im}$')
# ax.set_zlabel(r'$V(\phi)$')
plt.savefig('fp_polar3d.pdf', transparent=True)

#%%