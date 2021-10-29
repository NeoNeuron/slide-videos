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
    def __init__(self, ax, data, bins, range, mean=None, var=None ):
        self.ax = ax
        self.data = data
        self.bins = bins
        self.range = range
        edges = np.linspace(*range, bins+1)
        self.rects = self.ax.bar(
            (edges[1:]+edges[:-1])/2, np.zeros(bins), 
            width=edges[1]-edges[0], 
            color='#10505B')
        if mean is None or var is None:
            raise RuntimeError("empty mean or var")
        gauss = Gaussian(mean, var)
        self.ax.plot(edges, gauss(edges), ls='--', color='#B72C31', label='高斯模型估计')
        self.ax.set_ylim(0,gauss(edges).max()*1.2)
        self.ax.legend(loc=1, fontsize=14)
        self.ax.set_xlim(*range)
        self.ax.set_ylabel('概率密度')
        self.ax.set_xlabel('真实上座人数')
        self.ax.set_title(f'模拟测试次数 : {0:5d}')
        self.number_of_sample_list = [1,10,20,40,60,100,200,400,600,800,1000,1500,2000,3000,5000]
    
    def __call__(self, i):
        if i>= len(self.number_of_sample_list):
            idx = self.number_of_sample_list[-1]
        else:
            idx = self.number_of_sample_list[i]
        counts, _ = np.histogram(self.data[:idx], bins=self.bins, range=self.range, density=True)
        for rect, h in zip(self.rects, counts):
            rect.set_height(h)
        self.ax.set_title(f'模拟测试次数 : {idx:5d}')
        return self.rects

#%%
# Simulate Bernoulli random tests

my_distribution = boltzmann
my_dist_args = dict(
    lambda_=1.4,
    N = 19,
)
# my_distribution = bernoulli
# my_dist_args = dict(
#     p=0.9,
# )

n = 300 # number of accumulated samples
K = 5000 # number of random tests

# calculate the accumulate mean and variance
single_mean, single_var  = my_distribution.stats(**my_dist_args, moments='mv')
var_mean = n*single_mean
var_var = n*single_var

bins = int(K**0.5)
range = (var_mean-var_var**0.5*3,var_mean+var_var**0.5*3)

# generate sampling data
attendence = my_distribution.rvs(**my_dist_args, size=(K,n), random_state=0)
attendence = np.sum(attendence, axis=1, dtype=float)

# %%
# gauss = Gaussian(var_mean, var_var)
# counts, edges = np.histogram(attendence, bins=bins, range=range, density=True)
# plt.bar((edges[1:]+edges[:-1])/2, counts, width=edges[1]-edges[0], color='b', alpha=0.5)
# plt.plot(edges, gauss(edges), color='r')
# %%

fig, ax = plt.subplots(1,1,dpi=300, gridspec_kw=dict(left=0.15, right=0.95, bottom=0.15))
uh = UpdateHistogram(ax, attendence, bins, range, var_mean, var_var)
anim = FuncAnimation(fig, uh, frames=15, interval=800, blit=True)
anim.save('test_movie.mp4', dpi=300, )
# %%