#%%
from pathlib import Path
path = Path('./normal_2d/')
path.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal as mn
import matplotlib.cm as cm


# %%
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'


class UpdateFigure:
    def __init__(self, ax1, ax2):

        self.colors = dict(
            blue        = '#375492',
            green       = '#88E685',
            dark_green  = '#00683B',
            red         = '#93391E',
            pink        = '#E374B7',
            purple      = '#A268B4',
            black       = '#000000',
        )
        # ====================
        # config data
        # ====================
        self.mean = np.array([0.9/2,1.1/2])#np.zeros(2)
        self.cov = 0.25*np.eye(2)
        self.cov[1,0] = self.cov[0,1] = 0.25*0.95
        self.cov = self.cov/8
        #self.xx, self.yy = np.meshgrid(np.linspace(0,2,101), np.linspace(0,2,101))
        #xysurf = mn.pdf(np.dstack((self.xx,self.yy)), self.mean, self.cov)
        #self.vmin, self.vmax = 0, np.max(xysurf)
        np.random.seed(1)
        self.xx = np.random.multivariate_normal(self.mean, self.cov, (10000,))

        self.scatter_main = ax1.scatter([], [],alpha=0.7, c=self.colors['red'],s=4,marker='x')
        self.ax1 = ax1
        ax1.axis('scaled')
        ax1.set_xlim([0,1])
        ax1.set_ylim([0,1])
        xticks=[i*0.2 for i in range(6)]
        yticks=[i*0.2 for i in range(6)]
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_xlabel(r'$x_1$(CPU负载)',fontsize=18)
        ax1.set_ylabel(r'$x_2$(内存占用)',fontsize=18)


        self.n = n = 11
        _xx, _yy = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))
        self.x, self.y = _xx.ravel(), _yy.ravel()
        self.top = self.get_top([],np.linspace(0,1,n),np.linspace(0,1,n),np.zeros([n,n]))
        self.bar = ax2.bar3d(self.x-0.03,self.y-0.03,0,0.06,0.06,self.top.T.ravel(),shade=True,color=self.colors['blue'])
        ax2.set_xticks([i*0.5 for i in range(3)])
        ax2.set_yticks([i*0.5 for i in range(3)])
        #ax2.set_xlim([-0.05,2.05])
        #ax2.set_ylim([-0.05,2.05])
        ax2.set_xlim([-0.02,1.02])
        ax2.set_ylim([-0.02,1.02])
        ax2.set_zticks([i*10 for i in range(3)])
        ax2.set_zlim(0, 25)

        ax2.view_init(elev=50, azim=-115)

        #ax2.invert_xaxis()
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor('w')
        ax2.yaxis.pane.set_edgecolor('w')
        ax2.zaxis.pane.set_edgecolor('w')
        ax2.xaxis.set_rotate_label(False)
        ax2.yaxis.set_rotate_label(False)
        ax2.set_xlabel(r'$x_1$', fontsize=18, rotation=0)
        ax2.set_ylabel(r'$x_2$', fontsize=18, rotation=0)
        ax2.tick_params(axis='x', which='major', pad=0)
        ax2.tick_params(axis='y', which='major', pad=0)
        self.ax2 = ax2


    def get_top(self, xx, x, y, top):
        x0, x1 = np.min(x), np.max(x)
        y0, y1 = np.min(y), np.max(y)
        n, m = x.shape[0], y.shape[0]
        int_x = (x1-x0)/(n-1)
        int_y = (y1-y0)/(m-1)
        #top = np.zeros([n,m])
        #print(xx)
        for num in range(len(xx)):
            z = xx[num][0]
            i = int(((z-x0-(int_x/2))//int_x)+1)
            z = xx[num][1]
            j = int(((z-y0-(int_y/2))//int_y)+1)
            if(i>=0 and i<n and j>=0 and j<m):
                top[i,j] += 1
        #print(top)
        return top



    @staticmethod
    def _gauss(mean, sigma, x):
        return np.exp(-(x-mean)**2/(2*sigma**2))/2/np.pi/sigma


    def __call__(self, i):
        # print(i)

        if (i>0 and i<=200):

            if i==0:
                self.ax1.scatter(0.6, 1.4, alpha=1, c=self.colors['green'],s=50,marker='x')
                

                self.ax1.axvline(0.6, ymax=0.7, ls='--', color=self.colors['green'])
                self.ax1.axhline(1.4, xmax=0.3, ls='--', color=self.colors['green'])


            self.scatter_main.remove()
            self.scatter_main = self.ax1.scatter(self.xx[:i,0], self.xx[:i,1], alpha=1, c=self.colors['red'],s=50,marker='x',zorder=1)

            if i==200:
                print('surface_output')
                self.bar.remove()
                n = 201
                _xx, _yy = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))
                mean_=np.mean(self.xx[:i],axis=0)
                cov_ = np.cov(self.xx[:i].T)
                xysurf = mn.pdf(np.dstack((_xx,_yy)), mean_, cov_)*i/self.n/self.n
                self.vmin, self.vmax = 0, np.max(xysurf)

                # self.scatter_main.remove()
                # self.scatter_main = self.ax1.pcolormesh(_xx, _yy, xysurf, cmap='turbo',shading='auto')
                self.ax1.pcolormesh(_xx, _yy, xysurf, cmap='turbo',shading='auto',alpha=0.8,zorder=2)
                print(self.vmax)
                self.ax2.plot_surface(_xx, _yy, xysurf, cmap='turbo',
                       cstride=1, rstride=1, vmin=self.vmin, vmax=self.vmax,zorder=200)

                return [self.scatter_main]

            self.bar.remove()
            n = self.n
            self.top = self.get_top([self.xx[i]],np.linspace(0,1,n),np.linspace(0,1,n),self.top)

            cmap = cm.get_cmap('turbo')
            rgba = [cmap(k/20) for k in self.top.T.ravel()] 
            self.bar = self.ax2.bar3d(self.x-0.03,self.y-0.03,0,0.06,0.06,self.top.T.ravel(),shade=True,color=rgba,zorder=1)

        return [self.scatter_main]
            

fig = plt.figure(figsize=(8,4))
spec = gridspec.GridSpec(1, 2, 
    left=0.10, right=0.90, top=0.95, bottom=0.15,
    hspace=0.5,
    figure=fig)
#fig.set_tight_layout(True)
ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[0,1], projection='3d')
# create a figure updater
nframes=240
ud = UpdateFigure(ax1, ax2)
fig.savefig(path/'test_cpu_ram_2.pdf')
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save(path/'cpu2-addmesh-2.mp4', fps=40, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
fig.savefig(path/'cpu2-addmesh-2_finalshot.pdf')
# %%