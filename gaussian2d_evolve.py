#%%
from gaussian2d import *
fig = plt.figure(figsize=(10,5),dpi=400,)
spec = gridspec.GridSpec(1, 1, 
    left=0.10, right=0.55, top=0.95, bottom=0.05, 
    figure=fig)
ax2 = fig.add_subplot(spec[0], projection='3d')
spec = gridspec.GridSpec(2, 1, 
    left=0.6, right=0.90, top=0.9, bottom=0.2, 
    height_ratios=[1,5], hspace=0.3,
    figure=fig)
ax1 = fig.add_subplot(spec[0])
ax3 = fig.add_subplot(spec[1])
# create a figure updater
nframes=60
ud = UpdateFigure(ax1, ax2, ax3)
#%%
#! set target
eig, eigvec = np.linalg.eigh(np.array([[1,-1.6],[-1.6,4]]))
# print(eigvec@np.array([[1,-1.6],[-1.6,4]])@eigvec.T)
ud.set_target('stretch', np.diag(eig), nframes)
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
anim.save('2d_gaussian1.mp4', fps=20, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

ud.cov = np.diag(eig)
theta = np.arctan(eigvec[1,0]/eigvec[1,1])
ud.set_target('rotation', theta, nframes)
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
anim.save('2d_gaussian2.mp4', fps=20, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

ud.cov = np.array([[1,-1.6],[-1.6,4]])
ud.set_target('translation', np.ones(2), nframes)
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
anim.save('2d_gaussian3.mp4', fps=20, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%