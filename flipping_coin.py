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

coin_front_marker = gen_marker('yuan_front.svg',0, flip=True)
coin_back_marker = gen_marker('yuan_back.svg',0,)
#%%
# test marker
plt.scatter([0],[0],  s=10000, color='k', marker=coin_back_marker)
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

class UpdateDist:
    def __init__(self, ax, data, ax_main, n_in_each_frame):
        self.ax = ax
        self.n_in_each_frame = n_in_each_frame
        # self.coin_front = self.ax.scatter([-1],[0],  s=200, facecolor=[230./255,0./255,18./255,1], marker=coin_front_marker)
        # self.coin_back = self.ax.scatter([1],[0],  s=200, facecolor=[0,176./255,80./255,1], marker=coin_back_marker)
        self.coin_front = self.ax.scatter([0],[-1],s=8000, facecolor='k', marker=coin_front_marker)
        self.coin_back = self.ax.scatter([0],[1],  s=8000, facecolor='k', marker=coin_back_marker)
        self.ax.set_xlim(-0.8,1)
        self.ax.set_ylim(-2,2)
        self.ax.invert_yaxis()
        self.data = data
        self.cummean = np.cumsum(self.data)/np.arange(1, data.shape[0]+1)
        self.day_data = np.arange(1, data.shape[0]+1)

        # main plot:
        self.line_main, = ax_main.plot([], [], '-o',
            color='#006D87',
            markersize=3,
            markerfacecolor='none',
            markeredgewidth=1)
        ax_main.set_xlabel('抛硬币次数', fontsize=25)
        ax_main.set_ylabel('正面朝上概率', fontsize=25)
        # ax_main.grid(linestyle='--')

        # now determine nice limits by hand:
        ymax = np.max(np.fabs(self.cummean))
        ymin = np.min(np.fabs(self.cummean))
        xlim = (-1, (data.shape[0])/100)
        ylim = (ymin*0.95, ymax*1.05)

        self.xlim0 = xlim
        self.ylim0 = np.array([0.1,0.8])
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(self.ylim0)

        self.ax_main = ax_main
        self.ax_main.axhline(0.5, ls='--', color='r')
        self.bar = self.ax_main.bar([],[])

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            n_inc = 1000
            # update lines
            idx = self.n_in_each_frame[i]
            self.line_main.set_data(self.day_data[:idx], self.cummean[:idx])
            if idx < self.data.shape[0]:
                xlim = np.array(self.ax_main.get_xlim())
                # ylim = np.array(self.ax_main.get_ylim())
                if self.day_data[idx] >= xlim[1]:
                    xlim[1] += self.xlim0[1]-self.xlim0[0]
                    self.ax_main.set_xlim(xlim)
                    self.ax_main.set_ylim((self.ylim0-0.5)*20/np.sqrt(self.ax_main.get_xlim()[1])+0.5)
            # if i % 10 == 0:
            #     ylim = np.array(self.ax_main.get_ylim())
            #     self.ax_main.set_ylim((ylim-0.5)*.8+0.5)
            
            # update scatter facecolor
            if idx >= self.data.shape[0]:
                self.coin_front.set_color('k')
                self.coin_back.set_color('k')
            else:
                if self.data[idx]>0:
                    self.coin_front.set_color('g')
                    self.coin_back.set_color('k')
                else:
                    self.coin_front.set_color('k')
                    self.coin_back.set_color('g')
        return self.bar
# %%
my_distribution = bernoulli
my_dist_args = dict(
    p=0.5,
)

n = 120000 # number of accumulated samples

# calculate the accumulate mean and variance
single_mean, single_var  = my_distribution.stats(**my_dist_args, moments='mv')
# generate sampling data
flip_results = my_distribution.rvs(**my_dist_args, size=(n,), random_state=99239)

fig = plt.figure(figsize=(9,3),dpi=400)
spec1 = gridspec.GridSpec(ncols=1, nrows=1, left=0.01, right=0.16, top=0.98, bottom=0.12, figure=fig)
ax0 = fig.add_subplot(spec1[0])
ax0.axis('off')
spec2 = gridspec.GridSpec(ncols=1, nrows=1, left=0.30, right=0.98, top=0.95, bottom=0.25, figure=fig)
ax1 = fig.add_subplot(spec2[0])

factor = np.arange(20)
for i in range(10):
    if i == 0:
        n_in_each_frame = factor.copy()
    else:
        n_in_each_frame = np.append(n_in_each_frame, factor*10*(i+1)+n_in_each_frame[-1])

ud = UpdateDist(ax0, flip_results, ax1, n_in_each_frame)
anim = FuncAnimation(fig, ud, frames=n_in_each_frame.shape[0], interval=150, blit=True)
fname = "flipping_coin_movie.mp4"
anim.save(fname, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
# test bining sequence
factor = np.arange(20)
for i in range(10):
    if i == 0:
        n_in_each_frame = factor.copy()
    else:
        n_in_each_frame = np.append(n_in_each_frame, factor*10*(i+1)+n_in_each_frame[-1])
plt.plot(n_in_each_frame)
# %%
video = VideoFileClip(fname, audio=False)
video = video.subclip(0,video.duration)

video.to_videofile(fname.split('.')[0] + '_recompressed.mp4', fps=24)


# %%