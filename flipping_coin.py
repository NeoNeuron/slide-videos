# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.stats import binom, bernoulli, boltzmann
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
from moviepy.editor import *

def gen_marker(fname, rotation=180, flip=False):
    person_path, attributes = svg2paths(fname)
    att_str = ''
    for att in attributes:
        att_str += att['d'] + ','
    person_marker = parse_path(att_str)
    person_marker.vertices -= person_marker.vertices.mean(axis=0)
    person_marker = person_marker.transformed(mpl.transforms.Affine2D().rotate_deg(rotation))
    person_marker = person_marker.transformed(mpl.transforms.Affine2D().scale(-1,1))
    if flip:
        trans = mpl.transforms.Affine2D()
        trans.set_matrix([[-1,0,0],[0,1,0],[0,0,1]])
        person_marker = person_marker.transformed(trans)
    return person_marker

coin_front_marker = gen_marker('bitcoin_front.svg',0, flip=True)
coin_back_marker = gen_marker('bitcoin_back.svg',)
#%%
# test marker
plt.scatter([0],[0],  s=10000, color='k', marker=coin_front_marker)
plt.gca().axis('off')
#%%

class UpdateDist:
    def __init__(self, ax, data, ax_main):
        self.ax = ax
        xn, yn = 40, 30
        xx, yy = np.meshgrid(np.arange(xn), np.arange(yn))
        self.xx = xx.flatten()
        self.yy = yy.flatten()
        self.coin_front_data = [[],[]]
        self.coin_back_data = [[],[]]
        self.coin_front, = self.ax.plot([],[], lw=0, ms=20, markerfacecolor=[230./255,0./255,18./255,1], marker=coin_front_marker)
        self.coin_back, = self.ax.plot([],[],  lw=0, ms=20, markerfacecolor=[0,176./255,80./255,1], marker=coin_back_marker)
        self.ax.set_xlim(-1,xn)
        self.ax.set_ylim(-1,yn)
        self.ax.invert_yaxis()
        self.data = data
        self.cummean = np.cumsum(self.data)/np.arange(1, data.shape[0]+1)
        self.day_data = np.arange(1, data.shape[0]+1)

        # main plot:
        self.line_main, = ax_main.plot([], [], '-o',
            color='#006D87',
            markersize=6,
            markerfacecolor='none',
            markeredgewidth=2)
        ax_main.set_xlabel('抛硬币次数', fontsize=40)
        ax_main.set_ylabel('正面朝上概率', fontsize=40)
        # ax_main.grid(linestyle='--')

        # now determine nice limits by hand:
        ymax = np.max(np.fabs(self.cummean))
        ymin = np.min(np.fabs(self.cummean))
        xlim = (-1, (data.shape[0])/10)
        ylim = (ymin*0.95, ymax*1.05)

        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)

        self.ax_main = ax_main
        self.ax_main.axhline(0.5, ls='--', color='r')
        self.bar = self.ax_main.bar([],[])

        


    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            n_inc = 10
            # update lines
            idx = (i)*n_inc
            self.line_main.set_data(self.day_data[:idx], self.cummean[:idx])
            if idx < self.data.shape[0]:
                xlim = np.array(self.ax_main.get_xlim())
                ylim = np.array(self.ax_main.get_ylim())
                if self.day_data[idx] >= xlim[1]:
                    self.ax_main.set_xlim(xlim+xlim[1]-xlim[0])
                    self.ax_main.set_ylim((ylim-0.5)*.9+0.5)
            # if i % 10 == 0:
            #     ylim = np.array(self.ax_main.get_ylim())
            #     self.ax_main.set_ylim((ylim-0.5)*.8+0.5)
            
            # update scatter facecolor
            for j in np.arange(n_inc):
                # idx = i-1
                idx = (i-1)*n_inc+j
                if idx >= self.data.shape[0]:
                    break
                if self.data[idx]>0:
                    self.coin_front_data[0].append(self.xx[idx])
                    self.coin_front_data[1].append(self.yy[idx])
                else:
                    self.coin_back_data[0].append(self.xx[idx])
                    self.coin_back_data[1].append(self.yy[idx])
            self.coin_front.set_data(self.coin_front_data[0], self.coin_front_data[1])
            self.coin_back.set_data(self.coin_back_data[0], self.coin_back_data[1])
        return self.bar
# %%
my_distribution = bernoulli
my_dist_args = dict(
    p=0.5,
)

n = 1200 # number of accumulated samples

# calculate the accumulate mean and variance
single_mean, single_var  = my_distribution.stats(**my_dist_args, moments='mv')
# generate sampling data
flip_results = my_distribution.rvs(**my_dist_args, size=(n,), random_state=99239)

fig, ax = plt.subplots(1,2,figsize=(30,10),dpi=100, gridspec_kw=dict(left=0.04, right=0.96, top=0.92, bottom=0.12))
ax[0].axis('off')

ud = UpdateDist(ax[0], flip_results, ax[1])
anim = FuncAnimation(fig, ud, frames=120, interval=100, blit=True)
fname = "flipping_coin_movie.mp4"
anim.save(fname, dpi=100, codec='mpeg4')
# %%

video = VideoFileClip(fname, audio=False)
video = video.subclip(0,video.duration)

video.to_videofile(fname, fps=24)


# %%