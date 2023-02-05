# 
# %%
from pathlib import Path
path = Path('./variance/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["axes.labelsize"] = 45
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def axesArrow(ax, xlim, ylim):
    ax.scatter(xlim[-1], ylim[0], s=180, color='k', marker='>', ).set_clip_on(False)
    ax.scatter(xlim[0], ylim[-1], s=180, color='k', marker='^', ).set_clip_on(False)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(5)
    return ax

def configAxes(ax, xlim, ylim):
    ax = axesArrow(ax, xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('$X$的观测次数', labelpad=15)
    ax.set_ylabel('观测值', labelpad=6)
    return ax

#%%
class dot_motion:
    def __init__(self, 
        ax:plt.Axes, n_dots:int, n_lines:int, duration:float, fps:int=24):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
        """

        self.ax = ax
        self.dot_timeline = np.zeros((n_dots, int(duration*fps)), dtype=object)
        self.dot_timeline[:]=None
        self.line_timeline = np.zeros((n_lines, int(duration*fps)), dtype=object)
        self.line_timeline[:]=None
        
        self.n_dots = n_dots
        self.n_lines = n_lines
        self.fps = fps
        self.dots = [None]*n_dots
        self.lines = [None]*n_lines


    def _draw_dot(self, x, y, c='r'):
        if c == 'r':
            dot, = self.ax.plot(x, y, 'o', ms=15, markeredgewidth=2.5, markerfacecolor='#C00000', markeredgecolor='#D77F66')
        elif c == 'b':
            dot, = self.ax.plot(x, y, 'o', ms=15, markeredgewidth=2.5, markerfacecolor='#108B96', markeredgecolor='#00B0F0')
        elif c == 'y':
            dot, = self.ax.plot(x, y, 'o', ms=15, markeredgewidth=2.5, markerfacecolor='#FFD966', markeredgecolor='#BF9000')
        else:
            raise AttributeError('Unsupported color')
        dot.set_clip_on(False)
        return dot

    def _draw_line(self, xs, ys, c='r'):
        if c == 'r':
            line, = self.ax.plot(xs, ys, zorder=-1, ls='--', lw=2, c='#C00000')
        elif c == 'b':
            line, = self.ax.plot(xs, ys, zorder=-1, ls='--', lw=2, c='#108B96')
        elif c == 'y':
            line, = self.ax.plot(xs, ys, zorder=-1, ls='--', lw=2, c='#FFD966')
        else:
            raise AttributeError('Unsupported color')
        line.set_clip_on(False)
        return line

    def register_dot_trajectory(self, idx, src, dist, start_time, duration=1, c='r'):
        self.dots[idx] = self._draw_dot([], [], c)
        if duration > 0:
            n_frames = int(duration*self.fps)
            positions = np.linspace(src, dist, n_frames+1)
            start_id = int(start_time*self.fps)
            for i in range(n_frames+1):
                self.dot_timeline[idx, start_id+i] = positions[i,:]
        elif duration == 0:
            start_id = int(start_time*self.fps)
            self.dot_timeline[idx, start_id] = src

    def register_line_trajectory(self, idx, src, dist, start_time, duration=1, c='r'):
        self.lines[idx] = self._draw_line([], [], c)
        if duration > 0:
            n_frames = int(duration*self.fps)
            positions = np.linspace(src, dist, n_frames+1)
            start_id = int(start_time*self.fps)
            for i in range(n_frames+1):
                self.line_timeline[idx, start_id+i] = positions[i,:]
        elif duration == 0:
            start_id = int(start_time*self.fps)
            self.line_timeline[idx, start_id] = src

    def __call__(self, i):
        if i>0 and i <= self.dot_timeline.shape[1]:
            for j in range(self.n_dots):
                if self.dot_timeline[j, i-1] is not None:
                    self.dots[j].set_data(*self.dot_timeline[j,i-1])
            for j in range(self.n_lines):
                if self.line_timeline[j, i-1] is not None:
                    self.lines[j].set_data(*self.line_timeline[j,i-1])
        return self.dots
#%%
fig, ax = plt.subplots(1,1, figsize=(8,5), gridspec_kw=dict(left=0.15, bottom=0.2) )
# draw axes arrow
xlim=[ 0.5, 6.5]
ylim=[-0.1, 2.1]
configAxes(ax, xlim, ylim)
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(5)
n_dots = 6
duration = 4
fps = 48
nframe = int(duration*fps + 1)
# create a figure updater
ud = dot_motion(ax, 6, 1, 6,fps=fps)
data=np.ones(6)
srcs = np.vstack((np.arange(1, n_dots+1), data)).T
dists = np.vstack((np.arange(1, n_dots+1), data+1)).T
for i in range(n_dots):
    ud.register_dot_trajectory(i, srcs[i], None, i*0.5, 0)
ud.register_line_trajectory(0, np.array([[xlim[0],xlim[0]],[1,1]]), np.array([xlim,[1,1]]), 3, 0.5)
anim = FuncAnimation(fig, ud, frames=nframe, blit=True)
anim.save(path/'D_zero.mp4', fps=fps, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
#%%
fig, ax = plt.subplots(1,1, figsize=(8,5), gridspec_kw=dict(left=0.15, bottom=0.2) )
xlim=[ 0.5, 6.5]
ylim=[-0.1, 2.1]
configAxes(ax, xlim, ylim)
n_dots = 6
n_lines = 1
duration = 7
fps = 48
nframe = int(duration*fps + 1)
# create a figure updater
np.random.seed(0)
data = np.random.rand(6)
srcs = np.vstack((np.arange(1, n_dots+1), data)).T
dists = np.vstack((np.arange(1, n_dots+1), data+1)).T
# create animiation
ud = dot_motion(ax, n_dots, n_lines, duration, fps)
for i in range(n_dots):
    ud.register_dot_trajectory(i, srcs[i], None, i*0.5, 0)
    ud.register_dot_trajectory(i, srcs[i], dists[i], 4, 1)
data_mean = data.mean()
ud.register_line_trajectory(
    0, np.array([[xlim[0],xlim[0]],[data_mean,data_mean]]), np.array([xlim,[data_mean,data_mean]]), 
    3, 0.5)
ud.register_line_trajectory(
    0, np.array([xlim,[data_mean,data_mean]]), np.array([xlim,[data_mean+1,data_mean+1]]), 
    5.5, 1)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframe, blit=True)
# save animation as *.mp4
anim.save(path/'D_plusC.mp4', fps=fps, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
#%%
fig, ax = plt.subplots(1,1, figsize=(8,8), gridspec_kw=dict(left=0.15, bottom=0.2) )
xlim=[ 0.5, 6.5]
ylim=[ 0.0, 2.5]
configAxes(ax, xlim, ylim)
n_dots = 6
n_lines = 2
duration = 6.1
fps = 48
nframe = int(duration*fps + 1)
# create a figure updater
np.random.seed(0)
data = np.random.rand(6)
srcs = np.vstack((np.arange(1, n_dots+1), data)).T
dists = np.vstack((np.arange(1, n_dots+1), data*3)).T
# create animiation
ud = dot_motion(ax, n_dots*3, n_lines, duration, fps)
for i in range(n_dots):
    ud.register_dot_trajectory(i, srcs[i], None, i*0.5, 0)
    ud.register_dot_trajectory(i+6, srcs[i], dists[i], 4, 1)
    ud.register_dot_trajectory(i+12, dists[i], None, 5, 0, c='b')
data_mean = data.mean()
ud.register_line_trajectory(
    0, np.array([[xlim[0],xlim[0]],[data_mean,data_mean]]), np.array([xlim,[data_mean,data_mean]]), 
    3, 0.5)
ud.register_line_trajectory(
    1, np.array([[xlim[0],xlim[0]],[data_mean*3,data_mean*3]]), np.array([xlim,[data_mean*3,data_mean*3]]), 
    5.5, 0.5, c='b')
anim = FuncAnimation(fig, ud, frames=nframe, blit=True)
anim.save(path/'D_timeC.mp4', fps=fps, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
fig, ax = plt.subplots(2,2, figsize=(16,12), gridspec_kw=dict(left=0.1, right=0.95, bottom=0.1, top=0.95, hspace=0.5) )
# draw axes arrow
xlim=[ 0.5, 6.5]
ylim=[-0.5, 1.5]
for axi in ax.flatten():
    configAxes(axi, xlim, ylim)
ax[1,0].set_xlabel('$Y$的观测次数', labelpad=15)
ax[0,1].set_xlabel('$X+Y$的观测次数', labelpad=15)
ax[1,1].set_xlabel('$X-Y$的观测次数', labelpad=15)
n_dots = 6
n_lines = 4
duration = 20
fps = 48
nframe = int(duration*fps + 1)
# create a figure updater
np.random.seed(0)
data1 = np.random.rand(6)
data2 = np.random.rand(6)
srcs1 = np.vstack((np.arange(1, n_dots+1), data1)).T
srcs2 = np.vstack((np.arange(1, n_dots+1), data2)).T
dists1 = np.vstack((np.arange(1, n_dots+1), data1+data2)).T
dists2 = np.vstack((np.arange(1, n_dots+1), data1-data2)).T
ud = dot_motion(ax[0,0], n_dots*8, n_lines, duration, fps)
ax[0,0].set_zorder(10)

ax00inv = ax[0,0].transData.inverted()
srcs2 = ax00inv.transform(ax[1,0].transData.transform(srcs2))
dists1 = ax00inv.transform(ax[0,1].transData.transform(dists1))
dists2 = ax00inv.transform(ax[1,1].transData.transform(dists2))
# ax[0,0].plot(dists1[:,0],dists1[:,1], 'o', ms=10)[0].set_clip_on(False)
# ax[0,0].plot(dists2[:,0],dists2[:,1], 'o', ms=10)[0].set_clip_on(False)
# create animiation
for i in range(n_dots):
    ud.register_dot_trajectory(i, srcs1[i], None, i*0.25, 0)
    ud.register_dot_trajectory(i+6, srcs2[i], None, 3+i*0.25, 0)
    ud.register_dot_trajectory(i+12, srcs1[i], dists1[i], 6+i, 0.9)
    ud.register_dot_trajectory(i+18, srcs2[i], dists1[i], 6+i, 0.9)
    ud.register_dot_trajectory(i+24, dists1[i], None, 6.9+i, 0, c='b')
    ud.register_dot_trajectory(i+30, srcs1[i], dists2[i], 13+i, 0.9)
    ud.register_dot_trajectory(i+36, srcs2[i], dists2[i], 13+i, 0.9)
    ud.register_dot_trajectory(i+42, dists2[i], None, 13.9+i, 0, c='y')
# data_mean = data.mean()
line_dests = np.array(
    [[xlim, [data1.mean(), data1.mean()]],
     [xlim, [data2.mean(), data2.mean()]],
     [xlim, [data1.mean()+data2.mean(), data1.mean()+data2.mean()]],
     [xlim, [data1.mean()-data2.mean(), data1.mean()-data2.mean()]]])
start_t = [2, 5, 12, 19]
colors = ['r', 'r', 'b', 'y']
for i, line, axi, s_t, c in zip(np.arange(4), line_dests, ax.T.flatten(), start_t, colors):
    line_start = line.copy()
    line_start[0,1]=line_start[0,0]
    line_start = ax00inv.transform(axi.transData.transform(line_start.T)).T
    line = ax00inv.transform(axi.transData.transform(line.T)).T
    ud.register_line_trajectory(i,
        line_start, line, 
        s_t, 0.5, c)
    
plt.savefig(path/'test.pdf')
anim = FuncAnimation(fig, ud, frames=nframe, blit=True)
anim.save(path/'D_add_minus.mp4', fps=fps, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
#%%