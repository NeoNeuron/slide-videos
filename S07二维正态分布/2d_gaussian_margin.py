#%%
from init import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import multivariate_normal as mn
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
plt.rcParams['xtick.labelsize']=22
plt.rcParams['ytick.labelsize']=22
# %%
class UpdateFigure:
    def __init__(self, ax1, ax2, ax3, mean=None, cov=None, y0=None):

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
        self.mean = np.zeros(2) if mean is None else mean
        self.cov = np.eye(2) if cov is None else cov
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
            color='k', fontsize=22, usetex=True)


        # ====================
        # plot 3d surface
        # ====================
        self.surf = ax2.plot_surface(self.xx, self.yy, xysurf, cmap=self.cm, alpha=0.4,
                        rstride=1, cstride=1, zorder=0, )# vmin=self.vmin, vmax=self.vmax)
        # ax2.view_init(10, None)
        self.y0 = -4 if y0 is None else y0
        verts1, verts2 = self.get_verts(self.y0, self.mean, self.cov)
        self.shade1 = Poly3DCollection([verts1], 
            facecolor=self.colors['red'], edgecolor='None', alpha=1.0, zorder=1) # Add a polygon instead of fill_between
        ax2.add_collection3d(self.shade1)
        self.shade2 = Poly3DCollection([verts2], 
            facecolor=self.colors['green'], edgecolor='None', alpha=0.7, zorder=1) # Add a polygon instead of fill_between
        ax2.add_collection3d(self.shade2)

        ax2.set_xlabel(r'$x$(CPU负载)', fontsize=20, labelpad=15)
        ax2.set_ylabel(r'$y$(内存占用)', fontsize=20, labelpad=15)
        ax2.tick_params(axis='x', which='major', pad=-2)
        ax2.tick_params(axis='y', which='major', pad=-2)
        ax2.tick_params(axis='z', which='major', pad=10)
        ax2.xaxis.set_rotate_label(True)
        ax2.yaxis.set_rotate_label(True)
        # ax2.zaxis.set_rotate_label(False)
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
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor('w')
        ax2.yaxis.pane.set_edgecolor('w')
        ax2.zaxis.pane.set_edgecolor('w')
        self.ax2 = ax2

        # ====================
        # draw conditional probability
        # ====================
        p_cond = np.array(verts1[:-2])
        self.line, = ax3.plot(p_cond[:,0], p_cond[:,2], lw=5, color=self.colors['blue'])
        self.shade_2d = ax3.fill_between(p_cond[:,0],0, p_cond[:,2], color=self.colors['red'], alpha=0.8)
        ax3.set_xlabel(r'$x$(CPU负载)', fontsize=20)
        ax3.text(-0.225,0.5,r'$f(x|\quad\qquad\qquad)$', fontsize=25,
            ha='center', va='center', rotation=90, color='k', transform=ax3.transAxes)
        ax3.set_ylabel(r'$y=%.2f$'%self.y0, color='red', fontsize=22, y=0.59)
        ax3.set_xticks(xticks)
        ax3.set_yticks([0,0.1,0.2])
        ax3.set_xlim(xticks[0], xticks[-1])
        ax3.set_ylim(zticks[0], zticks[-1])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        self.ax3 = ax3

    @staticmethod
    def get_verts(yp_, mean_, cov_):
        func = lambda x: mn.pdf((x,yp_), mean_, cov_)
        x_array = np.linspace(-4,4,201)
        verts1 = [(xi, yp_, func(xi)) for xi in x_array] \
                + [(4, yp_, 0), (-4, yp_, 0)]
        verts2 = [(xi, yp_, func(xi)) for xi in x_array] \
                + [(4, yp_, 0.2), (-4, yp_, 0.2)]
        return verts1, verts2

    @staticmethod
    def gen_text(_mean, _cov):
        return r"$\begin{matrix}\mu_1=%.1f\\\mu_2=%.1f\end{matrix}\quad\begin{matrix}\sigma_1=%.1f \\ \sigma_2=%.1f\end{matrix}\quad\rho=%.1f$"%(
                    *_mean, _cov[0,0], _cov[1,1], _cov[0,1]/np.sqrt(_cov[0,0]*_cov[1,1]))

    def __call__(self, i):
        if i==0:
            i = 90
        mean_ = self.mean
        cov_ = self.cov.copy()
        y_ = self.y0 + i/30
        # update tex math
        self.tex.set_text(self.gen_text(mean_, cov_))
        # update 3d shades
        verts1, verts2 = self.get_verts(y_, mean_, cov_)
        self.shade1.set_verts([verts1])
        self.shade2.set_verts([verts2])
        # update 2d line
        p_cond = np.array(verts1[:-2])
        self.line.set_data(p_cond[:,0], p_cond[:,2])
        ax3.set_ylabel(r'$y=%.2f$'%y_, color='red', fontsize=22, y=0.59)
        # update shade
        verts_2d = [[v[0], v[2]] for v in verts1]
        self.shade_2d.set_verts([verts_2d])
        if np.abs(y_-0.7)<1e-4:
            plt.savefig(path/'test_margin.pdf')

        return [self.line,]

# ====================
# create canvas and config axis layouts
# ====================
def create_canvas():
    fig = plt.figure(figsize=(10,4.5))
    spec = fig.add_gridspec(1, 1, 
    left=-0.05, right=0.50, top=1.05, bottom=0.08, 
    figure=fig)
    ax2 = fig.add_subplot(spec[0], projection='3d')
    spec = fig.add_gridspec(1, 1, 
    left=0.65, right=0.98, top=0.71, bottom=0.20, 
    figure=fig)
    ax3 = fig.add_subplot(spec[0])
    spec = fig.add_gridspec(1, 1, 
    left=0.62, right=0.92, top=0.92, bottom=0.82, 
    figure=fig)
    ax1 = fig.add_subplot(spec[0])
    return fig, ax1, ax2, ax3

fig, ax1, ax2, ax3 = create_canvas()
# create a figure updater
nframes=240
cov = np.array([[1,-0.5],[-0.5,1]])
ud = UpdateFigure(ax1, ax2, ax3, cov=cov)
plt.savefig(path/'test_margin1.pdf')
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save(path/'2d_gaussian_margin.mp4', fps=24, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%