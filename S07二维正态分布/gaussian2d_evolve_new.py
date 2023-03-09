#%%
from init import *
from gaussian2d import *
class Update_mu1(UpdateFigure):
    def __init__(self, ax1, ax2, ax3):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(50)
    
    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\mu}_1=%.1f$"%(_mean[0])

class Update_mu2(UpdateFigure):
    def __init__(self, ax1, ax2, ax3):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(50)
    
    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\mu}_2=%.1f$"%(_mean[1])

class Update_sigma1(UpdateFigure):
    def __init__(self, ax1, ax2, ax3):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(50)
    
    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\sigma}_1^2=%.1f$"%(_cov[0,0])

class Update_sigma2(UpdateFigure):
    def __init__(self, ax1, ax2, ax3):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(50)
    
    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\sigma}_2^2=%.1f$"%(_cov[1,1])

class Update_rho(UpdateFigure):
    def __init__(self, ax1, ax2, ax3, ax4, data):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(30)
        self.data = data
        x, y = self.gen_sample(self.cov[0,1])
        self.scatter, = ax4.plot(x,y,'o', alpha=0.5, ms=5,
                                 mec='#353A71', mfc='#A7BED2')

    def gen_sample(self, rho):
        Y = self.data[1,]*np.sqrt(1-np.abs(rho))+self.data[2]*np.sqrt(np.abs(rho))
        if rho >= 0:
            X = self.data[0,]*np.sqrt(1-np.abs(rho))+self.data[2]*np.sqrt(np.abs(rho))
        else:
            X = self.data[0,]*np.sqrt(1-np.abs(rho))-self.data[2]*np.sqrt(np.abs(rho))
        return X, Y
    
    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\rho}=%.1f$"%(_cov[0,1])

    def __call__(self, i):
        ret = super().__call__(i)
        if self.trans_type == 'morph':
            mean_ = self.mean.copy()
            cov_ = self.cov+self.diff*i
            x, y = self.gen_sample(cov_[0,1])
            self.scatter.set_data(x, y)
        return ret

def create_canvas_horizontal():
    fig = plt.figure(figsize=(13,4))
    spec = gridspec.GridSpec(1, 1, 
        left=-0.05, right=0.45, top=1.20, bottom=0.00, 
        figure=fig)
    ax2 = fig.add_subplot(spec[0], projection='3d')
    spec = gridspec.GridSpec(1, 1, 
        left=0.45, right=0.65, top=0.88, bottom=0.20, 
        figure=fig)
    ax3 = fig.add_subplot(spec[0])
    spec = gridspec.GridSpec(1, 1, 
        left=0.5, right=0.88, top=1.0, bottom=0.9, 
        figure=fig)
    ax1 = fig.add_subplot(spec[0])
    spec = gridspec.GridSpec(1, 1, 
        left=0.72, right=0.92, top=0.88, bottom=0.20, 
        figure=fig)
    ax4 = fig.add_subplot(spec[0])
    return fig,ax1,ax2,ax3, ax4

# %%
fig, ax1, ax2, ax3 = create_canvas_vertical()
# create a figure updater
nframes=60
ud = Update_mu1(ax1, ax2, ax3)
ud.cov = np.array([[1,0],[0,1]])
ud.mean = np.array([-2,0],)
ud.set_target('translation', np.array([4,0]),nframes)
plt.savefig(path/'test.pdf')
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save(path/'2d_gaussian_mu1.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
#%%
fig, ax1, ax2, ax3 = create_canvas_vertical()
# create a figure updater
nframes=60
ud = Update_sigma1(ax1, ax2, ax3)
ud.cov = np.array([[1,0],[0,1]])
ud.mean = np.array([0,0],)
ud.set_target('stretch', np.diag([4,1]),nframes)
# plt.savefig('test.pdf')
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save(path/'2d_gaussian_sigma1.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
plt.close('all')
# %%
fig, ax1, ax2, ax3 = create_canvas_vertical()
# create a figure updater
nframes=60
ud = Update_mu2(ax1, ax2, ax3)
ud.cov = np.array([[1,0],[0,1]])
ud.mean = np.array([0,-2],)
ud.set_target('translation', np.array([0,4]),nframes)
# plt.savefig('test.pdf')
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save(path/'2d_gaussian_mu2.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
plt.close('all')
#%%
fig, ax1, ax2, ax3 = create_canvas_vertical()
# create a figure updater
nframes=60
ud = Update_sigma2(ax1, ax2, ax3)
ud.cov = np.array([[1,0],[0,1]])
ud.mean = np.array([0,0],)
ud.set_target('stretch', np.diag([1,4]),nframes)
# plt.savefig('test.pdf')
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save(path/'2d_gaussian_sigma2.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
plt.close('all')
#%%
fig, ax1, ax2, ax3, ax4 = create_canvas_horizontal()
data = np.random.randn(3,400)
nframes=60
ud = Update_rho(ax1, ax2, ax3, ax4, data)
ud.format2d(ax4, 30, 14)
ud.format2d(ud.ax3, 30, 14)
ud.cov = np.array([[1,0.8],[0.8,1]])
ud.mean = np.array([0,0],)
ud.set_target('morph', np.array([[0,-1.6], [-1.6,0]]),nframes)
ud.ax2.set_zlim(0,0.3)
ud.ax2.set_zticks([0,0.1,0.2,0.3])
ud.ax2.set_zticks(np.arange(7)*5e-2, minor=True)
for axis in ('x','y','z'):
    ud.ax2.tick_params(axis=axis, labelsize=14)
plt.savefig(path/'test.pdf')
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save(path/'2d_gaussian_rho.mp4', fps=20, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
plt.close('all')
# %%
class Update_multiple(UpdateFigure):
    def __init__(self, ax1, ax2, ax3):
        super().__init__(ax1, ax2, ax3)
        self.tex.set_fontsize(26)

    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\begin{matrix}\mu_1=%.1f\\\mu_2=%.1f\end{matrix}\quad\begin{matrix}\sigma_1=%.1f \\ \sigma_2=%.1f\end{matrix}\quad\rho=%.1f$"%(
                    *_mean, _cov[0,0], _cov[1,1], _cov[0,1]/np.sqrt(_cov[0,0]*_cov[1,1]))

fig, ax1, ax2, ax3 = create_canvas_vertical()
nframes=60
ud = Update_multiple(ax1, ax2, ax3)
#%%
# initialize mean and cov
ud.mean = np.zeros(2)
ud.cov = np.eye(2)
# initialize mean and cov
ud.set_target('translation', np.array([-1,2]),nframes)
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
anim.save(path/'2d_gaussian_multi_phase1.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# change to new init state
ud.mean += np.array([-1,2])
ud.set_target('stretch', np.diag([2,0.5]),nframes)
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
anim.save(path/'2d_gaussian_multi_phase2.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# change to new init state
ud.cov *= np.diag([2,0.5])
ud.set_target('morph', np.array([[0,-0.8],[-0.8,0]]),nframes)
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
anim.save(path/'2d_gaussian_multi_phase3.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# save animation as *.mp4
# %%