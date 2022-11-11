# %%
from pathlib import Path
path = Path('central_limit_theorem/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.stats import bernoulli, uniform, norm
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24

def gen_marker(fname, rotation=180):
    person_path, attributes = svg2paths(fname)
    person_marker = parse_path(attributes[0]['d'])
    person_marker.vertices -= person_marker.vertices.mean(axis=0)
    person_marker = person_marker.transformed(mpl.transforms.Affine2D().rotate_deg(rotation))
    person_marker = person_marker.transformed(mpl.transforms.Affine2D().scale(-1,1))
    return person_marker

flight_marker = gen_marker('icons/flight.svg',0)
#%%
# test marker
xx, yy = np.meshgrid(np.arange(10), np.arange(10))
fig, ax = plt.subplots(1,1, figsize=(10,10), dpi=150)
ax.scatter(xx.flatten(),yy.flatten(),  s=2000, color='k', marker=flight_marker )
ax.axis('off')
#%%

class UpdateDist:
    def __init__(self, ax, data, ax_main, ax_right, ax_top, days, threshold):
        self.ax = ax
        self.threshold = threshold
        xn, yn = 41, 9
        xx, yy = np.meshgrid(np.arange(xn), np.arange(yn))
        self.sc_flight = self.ax.scatter(xx.flatten()[:days],yy.flatten()[:days], s=2000, facecolor=[0,0,0,1], marker=flight_marker)
        self.color = np.tile([0,0,0,1],(int(days),1)).astype(float)
        self.ax.set_xlim(-1,xn)
        self.ax.set_ylim(-1,yn)
        self.ax.invert_yaxis()
        self.data = data
        self.day_data = np.arange(data.shape[0])+1

        # main plot:
        self.line_main, = ax_main.plot([], [], 'o',
            markerfacecolor=np.array([0,109,135,100])/255.0, #006D87
            markeredgecolor=np.array([0,109,135,255])/255.0, #006D87
            markersize=10,
            markeredgewidth=2)
        ax_main.set_xlabel('日期', fontsize=40)
        ax_main.set_ylabel('总利润(万)', fontsize=35)
        # ax_main.grid(linestyle='--')

        # now determine nice limits by hand:
        ymax = np.max(np.fabs(data))
        ymin = np.min(np.fabs(data))
        xlim = (-1, data.shape[0]+1)
        ylim = (int(ymin*0.95), int(ymax*1.05))

        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)

        self.ax_main = ax_main

        self.bins = int(ylim[1]-ylim[0] + 1)
        self.range = [ylim[0]-0.5, ylim[1]+0.5]
        counts, edges = np.histogram(data, range=self.range, bins = self.bins)
        self.rects = ax_right.barh((edges[1:]+edges[:-1])/2, np.zeros_like(counts), color='#006D87')
        self.data_line_top = (np.cumsum(data>=self.threshold)/(np.arange(data.shape[0])+1))*100
        self.line_top, = ax_top.plot([],[], color='#006D87', lw=5)
        ax_top.set_ylabel('利润达\n预期概率(%)', fontsize=35)
        ax_right.set_xlabel('天数', fontsize=40)

        ax_top.set_xlim(self.ax_main.get_xlim())
        ax_top.set_ylim(89.9,100.1)
        ax_right.set_ylim(self.ax_main.get_ylim())
        ax_right.set_xlim(0,36)
        ax_main.axhline(135,  color='r', ls='--', lw=2)
        ax_right.axhline(135, color='r', ls='--', lw=2)
        ax_main.set_yticklabels(ax_main.get_yticks()*0.1)

        # fit the distribution with gaussian
        func = lambda x,b,c: np.exp(-(x-b)**2/2/c)
            
        y_grid = np.linspace(ylim[0], ylim[1], 200)
        # gauss_curve = func(y_grid, 0.9*seats, 0.09*seats)
        gauss_curve = func(y_grid, 0.45*seats, (0.3-0.45**2)*seats)
        gauss_curve /= np.sum(gauss_curve*(y_grid[1]-y_grid[0]))
        gauss_curve *= days
        ax_right.plot(gauss_curve, y_grid, color='#B15A43', lw=5)

        self.ax_top= ax_top
        self.ax_right= ax_right

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            n_inc = 5
            # update lines
            idx = (i)*n_inc
            self.line_main.set_data(self.day_data[:idx], self.data[:idx])
            self.line_top.set_data(self.day_data[:idx], self.data_line_top[:idx])
            
            counts, _ = np.histogram(self.data[:idx], range=self.range, bins = self.bins)
            for rect, h in zip(self.rects, counts):
                rect.set_width(h)
            # update scatter facecolor
            for j in np.arange(n_inc):
                # idx = i-1
                idx = (i-1)*n_inc+j
                if idx < self.data.shape[0]:
                    self.color[idx,:] = [230./255,0./255,18./255,1] if self.data[idx]<self.threshold else [0,176./255,80./255,1]
            self.sc_flight.set_facecolor(self.color)
        return self.rects
# %%
ticket_price = 500
my_dist_args = {
        'bernoulli': {
            'gen': bernoulli,
            'pm': {
                'p':0.9,
             },
            'range': (-350,350),
            'ylim': (0, 0.4),
        },
        'uniform': {
            'gen': uniform,
            'pm': {
                'scale':1
             },
            'range': (0,500),
            'ylim': (0, 0.4),
        },
        'norm': {
            'gen': norm,
            'pm': {
                'loc':0.45*319,
                'scale':np.sqrt((0.3-0.45**2)*319)
             },
            'range': (-350,350),
            'ylim': (0, 0.6),
        },
    }

n = 350 # number of accumulated samples
K = 100000 # number of random tests

threshold = my_dist_args['norm']['gen'].ppf(0.06, **my_dist_args['norm']['pm'])
print(threshold)

n = 350 # number of accumulated samples
K = 100000 # number of random tests
zscore=False
# generate sampling data
uniform_rvs = my_dist_args['uniform']['gen'].rvs(**my_dist_args['uniform']['pm'], size=(K,n), random_state=1240)
bernoulli_rvs = my_dist_args['bernoulli']['gen'].rvs(**my_dist_args['bernoulli']['pm'], size=(K,n), random_state=12)
attendence = uniform_rvs*bernoulli_rvs

days = 365
seats = 319
margin = np.sum(attendence[100:100+days,:seats], axis=1, dtype=float)

from matplotlib.ticker import NullFormatter 
nullfmt = NullFormatter()         # no labels

fig = plt.figure(figsize=(30,15),dpi=200)
spec1 = gridspec.GridSpec(ncols=1, nrows=1, left=0.00, right=1.00, top=1.00, bottom=0.58, figure=fig)
ax0 = fig.add_subplot(spec1[0])
ax0.axis('off')

# definitions for the axes
left, width = 0.08, 0.75
bottom, height = 0.08, 0.27
left_h = left + width + 0.02
bottom_h = bottom + height + 0.02
rect_main = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.12, height]

axMain = plt.axes(rect_main)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

ud = UpdateDist(ax0, margin, axMain, axHisty, axHistx, days, threshold)
anim = FuncAnimation(fig, ud, frames=84, blit=True)
anim.save(path/'flight_movie_money.mp4', fps=12, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%