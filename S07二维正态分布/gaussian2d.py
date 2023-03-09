#%%
from init import *
from scipy.stats import multivariate_normal as mn
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
# %%
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
            ha='center', va='center',
            color='k', fontsize=25, usetex=True)

        # ====================
        # plot 3d surface
        # ====================
        self.surf = ax2.plot_surface(self.xx, self.yy, xysurf, cmap=self.cm,
                        rstride=1, cstride=1, vmin=self.vmin, vmax=self.vmax)
        ax2.view_init(10, None)

        ax2.set_xlabel(r'$x$', fontsize=40, labelpad=15)
        ax2.set_ylabel(r'$y$', fontsize=40, labelpad=15)
        ax2.tick_params(axis='x', which='major', pad=-2)
        ax2.tick_params(axis='y', which='major', pad=-2)
        ax2.tick_params(axis='z', which='major', pad=10)
        ax2.zaxis.set_rotate_label(False)
        # ax2.set_zlabel(r'$f(y)$', rotation=0, fontsize=20)
        xticks=np.linspace(-4,4,9,dtype=int)
        yticks=np.linspace(-4,4,9,dtype=int)
        zticks=np.arange(5)*5e-2
        ax2.set_xticks(xticks,minor=True)
        ax2.set_yticks(yticks,minor=True)
        ax2.set_zticks(zticks,minor=True)
        ax2.set_xticks([-3,0,3],)
        ax2.set_yticks([-3,0,3],)
        ax2.set_zticks([0,.1,.2],)
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
        self.mesh = ax3.pcolormesh(self.xx, self.yy, xysurf, cmap=self.cm, lw=0)
        self.format2d(ax3, labelsize=40, ticksize=20)
        self.ax3 = ax3

    @staticmethod
    def format2d(ax, labelsize, ticksize):
        ticks=np.linspace(-4,4,9,dtype=int)
        ax.axis('scaled')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_ticks(ticks, minor=True)
            axis.set_ticks([-3,0,3])
            axis.set_ticklabels([-3,0,3], fontsize=ticksize)
        ax.set_xlabel(r'$x$', fontsize=labelsize, labelpad=0)
        ax.set_ylabel(r'$y$', fontsize=labelsize, rotation=0, va='center', labelpad=10)
    
    def set_target(self, trans_type, diff, nframe):
        self.trans_type = trans_type
        if trans_type == 'stretch':
            self.diff = (diff-np.eye(2))*1.0/(nframe-1)
        else:
            self.diff = diff*1.0/(nframe-1)

    @staticmethod
    def rot(mat, theta):
        rot_mat =  np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        return rot_mat.T@mat@rot_mat

    @staticmethod
    def stretch(mat, diff):
        stretch_mat = np.sqrt(np.eye(2) + diff)
        return stretch_mat@mat@stretch_mat

    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\boldsymbol{\mu}=\begin{bmatrix}%.1f\\%.1f\end{bmatrix},\boldsymbol{\Sigma}=\begin{bmatrix}%.1f & %.1f \\ %.1f & %.1f\end{bmatrix}$"%(*_mean, *_cov.flatten())

    def __call__(self, i):
        if self.trans_type == 'rotation':
            mean_ = self.mean.copy()
            cov_ = self.rot(self.cov, self.diff*i)
        elif self.trans_type == 'stretch':
            mean_ = self.mean.copy()
            cov_ = self.stretch(self.cov, self.diff*i)
        elif self.trans_type == 'translation':
            mean_ = self.mean+self.diff*i
            cov_ = self.cov.copy()
        elif self.trans_type == 'morph':
            mean_ = self.mean.copy()
            cov_ = self.cov+self.diff*i

        xysurf = mn.pdf(np.dstack((self.xx,self.yy)), mean_, cov_)

        self.surf.remove()
        self.surf = self.ax2.plot_surface(self.xx, self.yy, xysurf, cmap=self.cm,
                        rstride=1, cstride=1, vmin=self.vmin, vmax=self.vmax)
        self.mesh.set_array(xysurf)
        self.tex.set_text(self.gen_text(mean_, cov_), )
        return [self.surf,]

def create_canvas_vertical():
    fig = plt.figure(figsize=(5,10),)
    spec = fig.add_gridspec(1, 1, 
        left=0.10, right=0.85, top=0.50, bottom=0.05, 
        figure=fig)
    ax3 = fig.add_subplot(spec[0], )
    spec = fig.add_gridspec(1, 1, 
        left=0.01, right=0.91, top=1.00, bottom=0.45, 
        figure=fig)
    ax2 = fig.add_subplot(spec[0], projection='3d')
    spec = fig.add_gridspec(1, 1, 
        left=0.10, right=0.90, top=1.00, bottom=0.90, 
        figure=fig)
    ax1 = fig.add_subplot(spec[0])
    return fig, ax1, ax2, ax3

# %%
if __name__ == '__main__':
    # %%
    fig, ax1, ax2, ax3 = create_canvas_vertical()
    # create a figure updater
    nframes=100
    ud = UpdateFigure(ax1, ax2, ax3)
    ud.cov = np.array([[1,-0.8],[-0.8,1]])
    ud.set_target('stretch', np.diag([1,4]),nframes)
    # user FuncAnimation to generate frames of animation
    plt.savefig(path/'test_gauss.pdf')
    plt.savefig(path/'test_gauss.png', dpi=300)
    anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
    # save animation as *.mp4
    anim.save(path/'2d_gaussian_stretch.mp4', fps=20, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
