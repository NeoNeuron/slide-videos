# %%
from pathlib import Path
path = Path('./moment_estimation/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.stats import bernoulli
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 18
plt.rcParams["xtick.labelsize"] = 40
plt.rcParams["ytick.labelsize"] = 40
plt.rcParams["axes.labelsize"] = 55
plt.rcParams["axes.labelpad"] = 15
plt.rcParams["xtick.major.pad"] = 10
# %% data
np.random.seed(1901)
mean,std=600,10 #均值，标准差
size=(200, 10)
data=np.random.randn(*size)*std+mean
data_mean=np.sum(data,axis=1)/10

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
        self.xlim = (0.95, 1.05)

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
        
        ylim = (0, 60)
        for axi in [ax1,ax2,ax3]:
            axi.set_xlim(self.xlim)
            axi.set_ylim(ylim)
            axi.set_xticks([0.95,1,1.05])
            axi.set_yticks([20,40,60])
        ax1.set_ylabel('统计个数', labelpad=10)
        ax1.set_xlabel('样本1阶矩/总体1阶矩')
        ax2.set_xlabel('样本2阶矩/总体2阶矩')
        ax3.set_xlabel('样本3阶矩/总体3阶矩')


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
#%%
fig = plt.figure(figsize=(28,10))
spec = gridspec.GridSpec(ncols=3, nrows=1, left=0.08, right=0.98, top=0.95, bottom=0.2, wspace=0.18, figure=fig)
ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])
ud = UpdateFigure(ax1, ax2, ax3,m1,m2,m3)
plt.savefig(path/'test.pdf')
anim = FuncAnimation(fig, ud, frames=120, blit=True)
#ax1.bar((intervals[1:]+intervals[:-1])/2, counts)
#%%
anim.save(path/'m123-2.mp4', fps=24, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
