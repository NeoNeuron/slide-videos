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

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 50
plt.rcParams["ytick.labelsize"] = 50



# %%
#数据
np.random.seed(1901)
mean,std=600,10 #均值，标准差
size=100000
data1=np.random.normal(mean,std,size)
#mean1,std1,size1=600,30,500
#data2=np.random.normal(mean1,std1,size1)
mask = np.zeros(10000).astype(bool)
data=data1.reshape(-1,10)
data_mean=np.sum(data,axis=1)/10
index=np.arange(1,10001,1)
#data_total=np.concatenate((data1,data2))
#np.random.shuffle(data_total)
#data_total.shape

# %%
m1 = data_mean/mean
data = data/mean
m2 = np.power(np.sum(data**2,axis=1)/10,1)*mean**2/(mean**2+std**2)
m3 = np.power(np.sum(data**3,axis=1)/10,1)*mean**3/(mean**3+3*mean*(std**2))
data = data*mean


# %%
m1.shape

# %%
print(np.min(m1),np.max(m1))
print(np.min(m2),np.max(m2))
print(np.min(m3),np.max(m3))

# %%
counts, intervals = np.histogram(m1,bins=10)

# %%
class UpdateFigure:
    def __init__(self, ax1, ax2, ax3,m1,m2,m3):

        self.colors = dict(
            flight_init=[0,0,0,1],
            main=np.array([0,109,135,255])/255.0, #006D87
            gauss=np.array([177,90,67,255])/255.0, #B15A43
            #flight_red=np.array([230,0,18,255])/255.0,
            #flight_green=np.array([0,176,80,255])/255.0,
        )
        
        self.data = [m1,m2,m3]

        # now determine nice limits by hand:
        ymax = np.max(np.fabs(self.data))
        ymin = np.min(np.fabs(self.data))
        ylim = (int(ymin*0.98), int(ymax*1.02))
        #print(ylim)
        #ylim = (int(ymin*0.98), int(ymax*1.02))
        ylim1 = (np.min(m1),np.max(m1))
        ylim2 = (np.min(m2),np.max(m2))
        ylim3 = (np.min(m3),np.max(m3))
        ylim1 = ylim2 = ylim3
        ax1.set_xlim(ylim1)
        ax2.set_xlim(ylim2) 
        ax3.set_xlim(ylim3)
        ax1.set_xticks([0.95,1,1.05])
        ax2.set_xticks([0.95,1,1.05])
        ax3.set_xticks([0.95,1,1.05])
        ax1.set_yticks([0,100,200])
        ax2.set_yticks([0,100,200])
        ax3.set_yticks([0,100,200])
        

        # initialize the bins of histogram
        self.bins = 50#int((ylim[1]-ylim[0] + 1)/2)
        self.range1 = [ylim1[0], ylim1[1]]
        self.width1 = (self.range1[1]-self.range1[0])/self.bins
        counts, edges = np.histogram(m1, range=self.range1, bins = self.bins)
        xmax = np.max(counts)
        self.rects1 = ax1.bar((edges[1:]+edges[:-1])/2, np.zeros_like(counts), width = self.width1,color=self.colors['main'],ec='k')


        self.range2 = [ylim2[0], ylim2[1]]
        self.width2 = (self.range2[1]-self.range2[0])/self.bins
        counts, edges = np.histogram(m2, range=self.range2, bins = self.bins)
        xmax = np.max([xmax, np.max(counts)])
        self.rects2 = ax2.bar((edges[1:]+edges[:-1])/2, np.zeros_like(counts),width = self.width2, color=self.colors['main'],ec='k')


        self.range3 = [ylim3[0], ylim3[1]]
        self.width3 = (self.range3[1]-self.range3[0])/self.bins
        counts, edges = np.histogram(m3, range=self.range3, bins = self.bins)
        xmax = np.max([xmax, np.max(counts)])
        self.rects3 = ax3.bar((edges[1:]+edges[:-1])/2, np.zeros_like(counts),width = self.width3, color=self.colors['main'],ec='k')
        
        #xlim = (0, int(xmax+5))
        xlim = (0, 220)
        #print(xlim)
        ax1.set_ylim(xlim)
        ax1.set_ylabel('统计个数',fontsize=80)
        ax2.set_ylim(xlim) 
        ax3.set_ylim(xlim)
        ax1.set_xlabel('样本1阶矩/总体1阶矩',fontsize=80)
        ax2.set_xlabel('样本2阶矩/总体2阶矩',fontsize=80)
        ax3.set_xlabel('样本3阶矩/总体3阶矩',fontsize=80)


    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            n_inc = 10
            # update lines
            idx = (i)*n_inc
            #self.line_main.set_data(self.nums[:idx], self.data[:idx])
            #self.line_top.set_data(self.nums[:idx], self.data_line_top[:idx])
            
            # update the height of bars for histogram
            counts, edges = np.histogram(self.data[0][:idx], range=self.range1, bins = self.bins)
            #print(counts)
            for rect, h in zip(self.rects1, counts):
                rect.set_height(h)

            counts, edges = np.histogram(self.data[1][:idx], range=self.range2, bins = self.bins)
            for rect, h in zip(self.rects2, counts):
                rect.set_height(h)

            counts, edges = np.histogram(self.data[2][:idx], range=self.range3, bins = self.bins)
            for rect, h in zip(self.rects3, counts):
                rect.set_height(h)

        return self.rects1
#%%
fig = plt.figure(figsize=(45,15),dpi=400)
spec = gridspec.GridSpec(ncols=3, nrows=1, left=0.08, right=0.9, top=0.95, bottom=0.2, wspace=0.15, figure=fig)
ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])
ud = UpdateFigure(ax1, ax2, ax3,m1,m2,m3)
plt.savefig(path/'test.pdf')
anim = FuncAnimation(fig, ud, frames=120, blit=True)
#ax1.bar((intervals[1:]+intervals[:-1])/2, counts)
anim.save(path/'m123-2.mp4', fps=24, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
