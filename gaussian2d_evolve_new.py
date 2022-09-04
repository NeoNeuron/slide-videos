#%%
from gaussian2d import *
class Update_mu1(UpdateFigure):
    def __init__(self, ax1, ax2, ax3):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(70)
    
    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\mu}_1=%.1f$"%(_mean[0])

class Update_mu2(UpdateFigure):
    def __init__(self, ax1, ax2, ax3):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(70)
    
    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\mu}_2=%.1f$"%(_mean[1])

class Update_sigma1(UpdateFigure):
    def __init__(self, ax1, ax2, ax3):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(70)
    
    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\sigma}_1^2=%.1f$"%(_cov[0,0])

class Update_sigma2(UpdateFigure):
    def __init__(self, ax1, ax2, ax3):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(70)
    
    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\sigma}_2^2=%.1f$"%(_cov[1,1])

class Update_rho(UpdateFigure):
    def __init__(self, ax1, ax2, ax3):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(40)
    
    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\rho}=%.1f$"%(_cov[0,1])

def gen_canvas():
    fig = plt.figure(figsize=(5,10),dpi=200,)
    spec = gridspec.GridSpec(3, 1, 
        left=0.10, right=0.90, top=0.95, bottom=0.05, 
        hspace=0.0,
        height_ratios=[0.1,5,4],
        figure=fig)
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1], projection='3d')
    ax3 = fig.add_subplot(spec[2], )
    return fig, ax1, ax2, ax3
# %%
fig, ax1, ax2, ax3 = gen_canvas()
# create a figure updater
nframes=60
ud = Update_mu1(ax1, ax2, ax3)
ud.cov = np.array([[1,0],[0,1]])
ud.mean = np.array([-2,0],)
ud.set_target('translation', np.array([4,0]),nframes)
# plt.savefig('test.pdf')
# %
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save('2d_gaussian_mu1.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
#%%
fig, ax1, ax2, ax3 = gen_canvas()
# create a figure updater
nframes=60
ud = Update_sigma1(ax1, ax2, ax3)
ud.cov = np.array([[1,0],[0,1]])
ud.mean = np.array([0,0],)
ud.set_target('stretch', np.diag([4,1]),nframes)
# plt.savefig('test.pdf')
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save('2d_gaussian_sigma1.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
fig, ax1, ax2, ax3 = gen_canvas()
# create a figure updater
nframes=60
ud = Update_mu2(ax1, ax2, ax3)
ud.cov = np.array([[1,0],[0,1]])
ud.mean = np.array([0,-2],)
ud.set_target('translation', np.array([0,4]),nframes)
# plt.savefig('test.pdf')
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save('2d_gaussian_mu2.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
#%%
fig, ax1, ax2, ax3 = gen_canvas()
# create a figure updater
nframes=60
ud = Update_sigma2(ax1, ax2, ax3)
ud.cov = np.array([[1,0],[0,1]])
ud.mean = np.array([0,0],)
ud.set_target('stretch', np.diag([1,4]),nframes)
# plt.savefig('test.pdf')
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save('2d_gaussian_sigma2.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
#%%
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

nframes=60
ud = Update_rho(ax1, ax2, ax3)
ud.cov = np.array([[1,0.8],[0.8,1]])
ud.mean = np.array([0,0],)
ud.set_target('morph', np.array([[0,-1.6], [-1.6,0]]),nframes)
# plt.savefig('test.pdf')
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save('2d_gaussian_rho.mp4', fps=20, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

# %%
