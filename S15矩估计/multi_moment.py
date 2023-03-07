# %%
from pathlib import Path
path = Path(__file__).parents[1]/'videos/moment_estimation/'
path.mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
plt.rcParams["xtick.labelsize"] = 30
plt.rcParams["ytick.labelsize"] = 30
# %% data
np.random.seed(1901)
mean,std=600,np.sqrt(2380) #均值，标准差
size=(200, 10)
data=np.random.randn(*size)*std+mean
data_mean=np.mean(data,axis=1)

m1 = data.mean(1)/mean
m2 = ((data/mean)**2).mean(1)*mean**2/(mean**2+std**2)
m3 = ((data/mean)**3).mean(1)*mean**3/(mean**3+3*mean*(std**2))

print(np.min(m1),np.max(m1))
print(np.min(m2),np.max(m2))
print(np.min(m3),np.max(m3))

# %%
class UpdateFigure:
    def __init__(self, ax1, ax2, ax3,m1,m2,m3):

        self.colors = dict(
            flight_init=[0,0,0,1],
            main='#2F5597', #006D87
            gauss=np.array([177,90,67,255])/255.0, #B15A43
            #flight_red=np.array([230,0,18,255])/255.0,
            #flight_green=np.array([0,176,80,255])/255.0,
        )
        
        self.data = [m1,m2,m3]

        # now determine nice limits by hand:
        self.xlim = (0.8, 1.2)

        # initialize the bins of histogram
        self.bins = 25#int((ylim[1]-ylim[0] + 1)/2)
        self.width1 = (self.xlim[1]-self.xlim[0])/self.bins
        counts, edges = np.histogram(m1, range=self.xlim, bins = self.bins)
        xmax = np.max(counts)
        self.rects1 = ax1.bar((edges[1:]+edges[:-1])/2, np.zeros_like(counts), width = self.width1,color=self.colors['main'],ec='k')


        self.width2 = (self.xlim[1]-self.xlim[0])/self.bins
        counts, edges = np.histogram(m2, range=self.xlim, bins = self.bins)
        xmax = np.max([xmax, np.max(counts)])
        self.rects2 = ax2.bar((edges[1:]+edges[:-1])/2, np.zeros_like(counts),width = self.width2, color=self.colors['main'],ec='k')


        self.width3 = (self.xlim[1]-self.xlim[0])/self.bins
        counts, edges = np.histogram(m3, range=self.xlim, bins = self.bins)
        xmax = np.max([xmax, np.max(counts)])
        self.rects3 = ax3.bar((edges[1:]+edges[:-1])/2, np.zeros_like(counts),width = self.width3, color=self.colors['main'],ec='k')
        
        ylim = (0, 50)
        for axi in [ax1,ax2,ax3]:
            axi.set_xlim(self.xlim)
            axi.set_ylim(ylim)
            axi.set_xticks(np.linspace(*self.xlim, 3))
            axi.set_yticks(np.arange(5)*10+10)
        ax1.set_ylabel('统计个数', fontsize=50)
        ax1.set_xlabel('样本1阶矩/总体1阶矩', fontsize=50)
        ax2.set_xlabel('样本2阶矩/总体2阶矩', fontsize=50)
        ax3.set_xlabel('样本3阶矩/总体3阶矩', fontsize=50)


    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            n_inc = 2
            # update lines
            idx = (i)*n_inc
            # update the height of bars for histogram
            counts, edges = np.histogram(self.data[0][:idx], range=self.xlim, bins = self.bins)
            for rect, h in zip(self.rects1, counts):
                rect.set_height(h)

            counts, edges = np.histogram(self.data[1][:idx], range=self.xlim, bins = self.bins)
            for rect, h in zip(self.rects2, counts):
                rect.set_height(h)

            counts, edges = np.histogram(self.data[2][:idx], range=self.xlim, bins = self.bins)
            for rect, h in zip(self.rects3, counts):
                rect.set_height(h)

        return self.rects1

fig, ax = plt.subplots(
    1,3, figsize=(28,10), 
    gridspec_kw=dict(left=0.06, right=0.97, top=0.95, bottom=0.15, wspace=0.18))
ud = UpdateFigure(ax[0],ax[1],ax[2],m1,m2,m3)
anim = FuncAnimation(fig, ud, frames=120, blit=True)
anim.save(path/'moment123.mp4', fps=24, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
fig.savefig(path/'moment123_finalshot.pdf')
# %%
