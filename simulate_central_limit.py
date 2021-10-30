#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 20
from matplotlib.animation import FuncAnimation
from scipy.stats import binom, bernoulli, boltzmann

def Gaussian(mean, var):
    def _wraper(x):
        _buffer = np.exp(-(x-mean)**2/2/var)
        return _buffer/_buffer.sum()
    return _wraper
    
class UpdateHistogram():
    '''Update constrained historgram.
    
    Bin size of histogram is 1.
    '''
    def __init__(self, ax, data, bins=10, range=None, mean=None, var=None ):
        self.ax = ax
        self.data = data
        self.bins = bins
        if range is None:
            self.rects = self.ax.bar(
                np.arange(bins), np.zeros(bins), 
                width=1, 
                color='#10505B')
        else:
            self.range = range
            edges = np.linspace(*range, bins+1)
            self.rects = self.ax.bar(
                (edges[1:]+edges[:-1])/2, np.zeros(bins),
                width=edges[1]-edges[0],
                color='#10505B')
        # if mean is None or var is None:
        #     raise RuntimeError("empty mean or var")
        # gauss = Gaussian(mean, var)
        # self.ax.plot(edges, gauss(edges), ls='--', color='#B72C31', label='高斯模型估计')
        # self.ax.set_ylim(0,gauss(edges).max()*1.2)
        # self.ax.legend(loc=1, fontsize=14)
        # self.ax.set_xlim(*range)
        self.ax.set_ylim(0,1.1)
        self.ax.set_xlim(-0.5,301)
        self.ax.set_ylabel('概率密度(归一化)')
        self.ax.set_xlabel('真实上座人数')
        self.ax.set_title(f'售票数 : {0:5d}')
        self.number_of_sample_list = [1,2,5,10,15,30,60,100,140,180,200,220,240,260,300]
    
    def __call__(self, i):
        if i>= len(self.number_of_sample_list):
            idx = self.number_of_sample_list[-1]
        else:
            idx = self.number_of_sample_list[i]
        margin = np.sum(self.data[:,:idx], axis=1, dtype=float)
        m_range = (margin.min()-0.5, margin.max()+0.5)
        m_bins = int(m_range[1]-m_range[0])
        # if m_bins < self.bins:

        counts, edges = np.histogram(margin, bins=m_bins, range=m_range, density=True)
        # counts /= counts.max()
        if len(self.rects) > len(counts):
            counts = np.hstack((counts, np.zeros(len(self.rects) - len(counts))))
            edges = np.hstack((edges, np.zeros(len(self.rects) - len(counts))))
        for rect, h, x in zip(self.rects, counts, edges):
            rect.set_x(x)
            rect.set_height(h)
        if i == len(self.number_of_sample_list):
            gauss = Gaussian(margin.mean(), margin.std()**2)
            self.ax.plot(np.arange(301), gauss(np.arange(301)), ls='--', color='#B72C31', label='高斯模型估计')
            self.ax.legend(loc=2, fontsize=14)
        self.ax.set_title(f'售票数 : {idx:5d}')
        return self.rects

#%%
# Simulate Bernoulli random tests

my_distribution = boltzmann
# my_dist_args = dict(
#     lambda_=1.4,
#     N = 19,
# )
my_distribution = bernoulli
my_dist_args = dict(
    p=0.9,
)

n = 300 # number of accumulated samples
K = 100000 # number of random tests

# calculate the accumulate mean and variance
single_mean, single_var  = my_distribution.stats(**my_dist_args, moments='mv')
bins = n

# generate sampling data
attendence = my_distribution.rvs(**my_dist_args, size=(K,n), random_state=1240)

# %%

fig, ax = plt.subplots(1,1,dpi=300, gridspec_kw=dict(left=0.15, right=0.95, bottom=0.15))
uh = UpdateHistogram(ax, attendence, bins, None,)
anim = FuncAnimation(fig, uh, frames=16, interval=800, blit=True)
anim.save('evolving_bernoulli.mp4', dpi=300, )
# %%