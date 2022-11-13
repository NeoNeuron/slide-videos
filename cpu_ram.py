#%%
from pathlib import Path
path = Path('./normal_2d/')
path.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal as mn
# %%
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'


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
        ax1.set_xlabel(r'$x_1$'+'(CPU负载)',fontsize=18)
        ax1.set_ylabel(r'$x_2$'+'(内存占用)',fontsize=18)


        self.bins = 10
        self.range1 = [0, 1]
        self.width1 = (self.range1[1]-self.range1[0])/self.bins
        counts, edges = np.histogram(self.xx[:50,0], range=self.range1, bins = self.bins)
        print(np.max(counts))
        #exit()
        self.rects1 = ax2.bar((edges[1:]+edges[:-1])/2, np.zeros_like(counts), width = self.width1,color=self.colors['red'],ec='k')

        #self.scatter_2 = ax2.scatter([], [],alpha=0.7, c=self.colors['red'],s=4,marker='x')
        #self.line2 = ax2.plot(np.linspace(0,1,101) ,self._gauss(0.9/2, np.sqrt(self.cov[0,0]), np.linspace(0,1,101)), color=self.colors['blue'], zorder=1)
        self.ax2 = ax2
        ax2.set_xlim([0,1])
        ax2.set_ylim([0,0.3])
        ax2.set_xlabel(r'$x_1$(CPU负载)',fontsize=18)
        ax2.set_ylabel(r'$x_1$频率',fontsize=18)


        self.range2 = [0, 1]
        self.width2 = (self.range2[1]-self.range2[0])/self.bins
        counts, edges = np.histogram(self.xx[:50,1], range=self.range2, bins = self.bins)
        self.rects2 = ax3.bar((edges[1:]+edges[:-1])/2, np.zeros_like(counts),width = self.width2, color=self.colors['red'],ec='k')

        #self.scatter_3 = ax3.scatter([], [],alpha=0.7, c=self.colors['red'],s=4,marker='x')
        #self.line3 = ax3.plot(np.linspace(0,1,101) ,self._gauss(1.1/2, np.sqrt(self.cov[1,1]), np.linspace(0,1,101)), color=self.colors['blue'], zorder=1)
        self.ax3 = ax3
        ax3.set_xlim([0,1])
        ax3.set_ylim([0,0.3])
        ax3.set_xlabel(r'$x_2$(内存占用)',fontsize=18)
        ax3.set_ylabel(r'$x_2$频率',fontsize=18)
        #plt.show()


    @staticmethod
    def _gauss(mean, sigma, x):
        return np.exp(-(x-mean)**2/(2*sigma**2))/np.sqrt(2*np.pi)/sigma


    def __call__(self, i):

        if i>0 and i<=50:
            if(i>1):
                self.shadow1.remove()
                self.shadow2.remove()


            self.scatter_main.remove()
            self.scatter_main = self.ax1.scatter(self.xx[:i,0], self.xx[:i,1], alpha=1, c=self.colors['red'],s=50,marker='x')

            #self.rects1.remove()
            counts, edges = np.histogram(self.xx[:i,0], range=self.range1, bins = self.bins)
            counts = counts*0.02
            # self.rects1.remove()
            # self.rects1 = self.ax2.bar((edges[1:]+edges[:-1])/2, counts, width = self.width1,color=self.colors['red'],ec='k')
            for rect, h in zip(self.rects1, counts):
                rect.set_height(h)

            idx = np.int64(self.xx[i-1,0]//0.1)
            #print(counts[idx])
            self.shadow1 = self.ax2.bar(idx*0.1+0.05, 3-counts[idx],bottom=counts[idx], width = self.width1, alpha=0.7, color='grey')

            #self.rects2.remove()
            counts, edges = np.histogram(self.xx[:i,1], range=self.range1, bins = self.bins)
            counts = counts*0.02
            for rect, h in zip(self.rects2, counts):
                rect.set_height(h)

            idx = np.int64(self.xx[i-1,1]//0.1)
            self.shadow2 = self.ax3.bar(idx*0.1+0.05, 3-counts[idx],bottom=counts[idx], width = self.width2, alpha=0.7, color='grey')
        elif i==55:
            self.shadow1.remove()
            self.shadow2.remove()
            self.line2 = self.ax2.plot(np.linspace(0,1,101) ,0.1*self._gauss(0.9/2, np.sqrt(self.cov[0,0]), np.linspace(0,1,101)), color=self.colors['blue'], zorder=5)
            self.line3 = self.ax3.plot(np.linspace(0,1,101) ,0.1*self._gauss(1.1/2, np.sqrt(self.cov[1,1]), np.linspace(0,1,101)), color=self.colors['blue'], zorder=5)

        return [self.scatter_main]
            

fig = plt.figure(figsize=(8,4))
spec = gridspec.GridSpec(2, 2, 
    left=0.08, right=0.95, top=0.95, bottom=0.15,
    hspace=0.5,
    wspace=0.3,
    figure=fig)
#fig.set_tight_layout(True)
ax1 = fig.add_subplot(spec[0:2,1])
ax2 = fig.add_subplot(spec[0,0], )
ax3 = fig.add_subplot(spec[1,0], )
# create a figure updater
nframes=72
ud = UpdateFigure(ax1, ax2, ax3)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save(path/'cpu-hist-shadow-2.mp4', fps=12, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
fig.savefig(path/'cpu-hist-shadow-2_finalshot.pdf')
# %%