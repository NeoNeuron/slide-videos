#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 16
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
    def __init__(self, ax, data, range=(0,350), mean=None, var=None):
        self.ax = ax
        self.data = data
        self.bar_width=1
        if range is None:
            raise RuntimeError('Missing range parameter.')
        self.bins = int(range[1]-range[0] + 1)
        self.range = [range[0]-0.5, range[1]+0.5]
        self.edges = np.linspace(*self.range, self.bins+1)
        self.rects = self.ax.bar(
            (self.edges[1:]+self.edges[:-1])/2, np.zeros(self.bins),
            width = self.bar_width,
            color='#206864')
        # if mean is None or var is None:
        #     raise RuntimeError("empty mean or var")
        # gauss = Gaussian(mean, var)
        # self.ax.plot(edges, gauss(edges), ls='--', color='#B72C31', label='高斯模型估计')
        # self.ax.set_ylim(0,gauss(edges).max()*1.2)
        # self.ax.legend(loc=1, fontsize=14)
        # self.ax.set_xlim(*range)
        self.ax.set_ylim(0,1.1)
        self.ax.set_xlim(-0.5,51)
        self.ax.set_ylabel('概率密度')
        self.ax.set_xlabel('真实上座人数')
        self.ax.set_title(f'售票数 : {0:5d}', fontsize=20)
        self.number_of_sample_list = [1,2,3,4,5,8,12,18,28,43,65,99,151,230, 350]
        self.color = plt.cm.Oranges(0.8*np.arange(len(self.number_of_sample_list))/len(self.number_of_sample_list))
        self.lines = []

    def set_colors(self, color_set):
        self.color = color_set

    def set_frame_numbers(self, number_set):
        self.number_of_sample_list = number_set
    

    def __call__(self, i):
        if i>= len(self.number_of_sample_list):
            idx = self.number_of_sample_list[-1]
        else:
            idx = self.number_of_sample_list[i]
        margin = np.sum(self.data[:,:idx], axis=1, dtype=float)

        counts, edges = np.histogram(margin, bins=self.bins, range=self.range, density=True)
        # counts /= counts.max()
        for rect, h, x in zip(self.rects, counts, self.edges):
            rect.set_x(x)
            rect.set_height(h)
        if i >= 0 and i < len(self.number_of_sample_list):
            # self._draw_gauss(i-1)
            self._draw_conti_hist(counts, edges, i)
            self._recolor()
        if i < 5:
            self.ax.set_xlim(-0.5,11)
        elif i < 10:
            self.ax.set_xlim(-0.5,51)
        else:
            self.ax.set_xlim(-0.5,351)
        # dark red : '#B72C31'
        self.ax.set_title(f'售票数 : {idx:5d}', fontsize=20)
        return self.rects

    def _draw_gauss(self, i):
        margin = np.sum(self.data[:,:self.number_of_sample_list[i]], axis=1, dtype=float)
        gauss = Gaussian(margin.mean(), margin.std()**2)
        print(margin.mean(), margin.std()**2)
        line, = self.ax.plot(
            (self.edges[1:]+self.edges[:-1])/2, gauss((self.edges[1:]+self.edges[:-1])/2), 
            ls='--', color=self.color[i],
            label='高斯拟合')
        self.lines.append(line)

    def _draw_conti_hist(self, counts, edges, i):
        line, = self.ax.plot(
            (edges[1:]+edges[:-1])/2, counts, 
            ls='--', lw=1, color=self.color[i],
            label='连线')
        self.lines.append(line)

    def _recolor(self,):
        for line, c in zip(self.lines, self.color[-len(self.lines):]):
            line.set_color(c)

#%%
# Simulate Bernoulli random tests

# my_distribution = boltzmann
# my_dist_args = dict(
#     lambda_=1.4,
#     N = 19,
# )
my_distribution = bernoulli
my_dist_args = dict(
    p=0.9,
)

n = 350 # number of accumulated samples
K = 100000 # number of random tests

# calculate the accumulate mean and variance
single_mean, single_var  = my_distribution.stats(**my_dist_args, moments='mv')
# generate sampling data
attendence = my_distribution.rvs(**my_dist_args, size=(K,n), random_state=1240)

# %%

fig, ax = plt.subplots(1,1,dpi=300, gridspec_kw=dict(left=0.15, right=0.95, bottom=0.15))

uh = UpdateHistogram(ax, attendence, (0,n))
number_list = [1,2,3,4,5,8,12,18,28,43,65,99,151,230,350]
uh.set_frame_numbers = number_list
uh.set_colors = plt.cm.Oranges(0.8*np.arange(len(number_list)/len(number_list)))

anim = FuncAnimation(fig, uh, frames=16, interval=800, blit=True)
anim.save('evolving_bernoulli.mp4', dpi=300, )
# %%
# draw static plots
pm = {
    'small': {
        'idx':0,
        'xlim':(-1,3),
        'ylim':(0,1),
    },
    'median': {
        'idx':6,
        'xlim':(0,20),
        'ylim':(0,0.5),
    },
    'large': {
        'idx':14,
        'xlim':(280,350),
        'ylim':(0,0.1),
    },
}
for key, val in pm.items():
    fig, ax = plt.subplots(1,1,dpi=300, gridspec_kw=dict(left=0.15, right=0.95, bottom=0.15))
    uh = UpdateHistogram(ax, attendence, (0,n))
    idx = val['idx']
    uh(idx)
    uh.lines[0].set_alpha(0)
    uh._draw_gauss(idx)
    uh.lines[1].set_color('r')
    ax.set_xlim(*val['xlim'])
    ax.set_ylim(*val['ylim'])
    ax.grid()
    fig.savefig(f'n_{key:s}.pdf')
# %%
