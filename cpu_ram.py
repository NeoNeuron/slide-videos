#%%
from pathlib import Path
path = Path('./normal_2d/')
path.mkdir(exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal as mn
from scipy.stats import norm
# %%
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'


class scatter_anim:
    def __init__(self, ax, data):
        # ====================
        # config data
        # ====================
        self.data = data
        self.scatter_main, = ax.plot([], [],'o', alpha=0.5, ms=5,
                                     mec='#353A71', mfc='#A7BED2')
        self.ax = ax
        ax.axis('scaled')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_ticks([i*0.2 for i in range(6)])
        ax.set_xlabel(r'$x$'+'(CPU负载)',fontsize=18)
        ax.set_ylabel(r'$y$'+'(内存占用)',fontsize=18)
    
    def __call__(self, i):
        if i>0 and i<=50:
            self.scatter_main.set_data(self.data[:i*8,0], self.data[:i*8,1])
        return [self.scatter_main]

class scatter2hist(scatter_anim):
    def __init__(self, ax, ax1, ax2, data):
        super().__init__(ax, data)
        self.scatter_main.set_data(self.data[:,0], self.data[:,1])
        self.scatter_main.set_alpha(0.3)
        self.ax.plot(data[:,0], data[:,1],'o', alpha=0.5, ms=5,
                    mec='#353A71', mfc='#A7BED2')
        self.bins= 20
        self.range = (0,1)
        # self.counts_x = self.hist(self.data[:,0])
        # self.counts_y = self.hist(self.data[:,1])
        edges = np.arange(self.bins)*(self.range[1]-self.range[0])/self.bins+self.range[0]
        self.bar1 = ax1.barh(edges, np.zeros(self.bins), height=1./self.bins, align='edge', ec='#353A71', fc='#E6782F')
        self.bar2 = ax2.bar(edges, np.zeros(self.bins), width=1./self.bins, align='edge', ec='#353A71', fc='#E6782F')
        ax1.set_ylim(0,1)
        ax2.set_xlim(0,1)
        ax1.set_xlim(0,55)
        ax2.set_ylim(0,55)
        self.ax1, self.ax2 = ax1, ax2
    
    def __call__(self, i):
        dx = dy = 1./24
        # bias = 0.01
        # speed = dx*2*(1-self.data[:,1]) + bias
        if i > 0 and i <= int(1/dx)+12:
            new_x = self.data[:,0]+dx*i
            self.scatter_main.set_data(new_x, self.data[:,1])
            counts = self.hist(self.data[new_x>1, 1])
            for bar, count in zip(self.bar1, counts):
                bar.set_width(count)
        elif i > int(1/dx)+12 and i <=int(1/dx)*2+12:
            new_y = self.data[:,1]+dy*(i-int(1/dx)-12-1)
            self.scatter_main.set_data(self.data[:,0], new_y)
            counts = self.hist(self.data[new_y>1, 0])
            for bar, count in zip(self.bar2, counts):
                bar.set_height(count)
        elif i ==int(1/dx)*2+18:
            x_grid = y_grid = np.linspace(*self.range,100)
            self.ax1.plot(norm.pdf(y_grid, loc=0.55, scale=np.sqrt(1/32))*400/self.bins, y_grid, 'k')
            self.ax2.plot(x_grid, norm.pdf(x_grid, loc=0.45, scale=np.sqrt(1/32))*400/self.bins, 'k')
        return self.bar1
            
    def hist(self, x):
        counts, _ = np.histogram(x, bins=self.bins, range=self.range)
        return counts

        
#%%
fig, ax = plt.subplots(2, 1, figsize=(4,4), gridspec_kw=dict( 
    left=0.08, right=0.95, top=0.95, bottom=0.15,
    hspace=0.5, wspace=0.3))

mean = np.array([0.45,0.55])
cov = np.array([[1, 0.95], [0.95, 1]])/32.
np.random.seed(1)
data = np.random.multivariate_normal(mean, cov, (10000,))
data[12,:] = [0.3, 0.7]
for idx, label in enumerate(('CPU负载', '内存占用')):
    ax[idx].plot(np.arange(1,26), data[:25,idx], '-o',ms=5, mew=1., 
                 mec='#353A71', mfc='#A7BED2', clip_on=False)
    ax[idx].set_xlabel('时间', fontsize=18)
    ax[idx].set_ylabel(label, fontsize=18)
    ax[idx].set_ylim(0,1)
    ax[idx].set_yticks([0,0.5,1])
    ax[idx].set_xlim(1, 25)
fig.savefig(path/'cpu_ram_sample_trace.pdf')

fig, ax = plt.subplots(2, 1, figsize=(4.5,4), gridspec_kw=dict( 
    left=0.14, right=0.95, top=0.95, bottom=0.15,
    hspace=0.5, wspace=0.3))
data[12,:] = [0.38, 0.68]
for idx, label in enumerate((r'$x$(CPU负载)', r'$y$(内存占用)')):
    ax[idx].hist(data[0:400,idx], ec='#353A71', fc='#E6782F', range=(0,1), bins=20, clip_on=False)
    ax[idx].set_ylabel('频率', fontsize=18)
    ax[idx].set_xlabel(label, fontsize=18)
    ax[idx].set_xlim(0,1)
fig.savefig(path/'cpu_ram_hist.pdf')

#%%
nframes=60
fig, ax = plt.subplots(1, 1, figsize=(4.5,4), gridspec_kw=dict(
    left=0.05, right=0.98, top=0.95, bottom=0.15,))
# create a figure updater
data[12,:] = [0.38, 0.68]
ud = scatter_anim(ax, data)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save(path/'cpu_ram_scatter.mp4', fps=12, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
fig.savefig(path/'cpu_ram_scatter_finalshot.png', dpi=300)
# %%
nframes=72
gap = 0.015
left, bottom, height, width = 0.12, 0.10, 0.6, 0.6
fig, ax = plt.subplots(1, 1, figsize=(6,6), gridspec_kw=dict(
    left=left, right=left+width, top=bottom+height, bottom=bottom,))
ax1 = fig.add_subplot(
    fig.add_gridspec(1,1,
                    left=left+width+gap, right=left+width+gap+0.2, 
                    top=bottom+height, bottom=bottom,
                    )[0])
ax2 = fig.add_subplot(
    fig.add_gridspec(1,1,
                    left=left, right=left+width, 
                    top=bottom+height+gap+0.2, bottom=bottom+height+gap,
                    )[0])
for axis in ('top', 'bottom', 'left', 'right'):
    ax1.spines[axis].set_visible(False)
    ax2.spines[axis].set_visible(False)
ax1.spines['left'].set_visible(True)
ax1.set_xticks([])
ax1.set_yticklabels([])
ax2.spines['bottom'].set_visible(True)
ax2.set_yticks([])
ax2.set_xticklabels([])
# create a figure updater
data[12,:] = [0.38, 0.68]
ud = scatter2hist(ax, ax1, ax2, data[:400])
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=np.arange(0,36), blit=True)
# save animation as *.mp4
anim.save(path/'cpu_ram_scatter2hist_p1.mp4', fps=12, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

anim = FuncAnimation(fig, ud, frames=np.arange(36,nframes+1), blit=True)
# save animation as *.mp4
anim.save(path/'cpu_ram_scatter2hist_p2.mp4', fps=12, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
fig.savefig(path/'cpu_ram_scatter2hist_finalshot.pdf')
# %%