# %%
from pathlib import Path
path = Path(__file__).parents[1]/'videos/intervel_estimation/'
path.mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 30
plt.rcParams["ytick.labelsize"] = 30
plt.rcParams["axes.labelsize"] = 50
def gen_marker(fname:str, rotation:float=180):
    """Generate maker from svg image file.
    Args:
        fname (str): filename of svg image.
        rotation (int, optional): 
            degree of rotation of original images. Defaults to 180.
    Returns:
        Object of marker.
    """
    person_path, attributes = svg2paths(fname)
    person_marker = parse_path(attributes[0]['d'])
    person_marker.vertices -= person_marker.vertices.mean(axis=0)
    person_marker = person_marker.transformed(mpl.transforms.Affine2D().rotate_deg(rotation))
    person_marker = person_marker.transformed(mpl.transforms.Affine2D().scale(-1,1))
    return person_marker

car_marker = gen_marker(path.parents[1]/'icons/car.svg',)
# %% 
class UpdateFigures(object):
    def __init__(self, ax, data) -> None:
        self.ax = ax
        self.data = data
        self.grow_bars_duration = 12
        self.grow_line_duration = 3
        self.move_dot_duration  = 6
        self.data_mean = np.mean(data, axis=1)
        self.bars = self.ax[1].bar(x, np.zeros_like(x), width=0.8, fc='#A7BED9', ec='#353A71', lw=2)
        self.sample_set_counter = 0
        self.toggle = np.zeros(3, dtype=bool)   #(bars, line, dot)
        self.play_next = np.zeros(3, dtype=bool)   #(bars, line, dot)
        self.play_next[0] = True
        self.dh = None
        self.hmax=None
        self.line = None
        self.static_dot = None
        self.dline = None
        self.dot=None
        self.dxy=None
        self.xmax=None
        self.pause = 0

    def reset_bar(self):
        for rect in self.bars:
            rect.set_height(450)

    def register_grow_bars(self, data, nframes):
        self.dh = (data-450)/nframes
        self.hmax = data
        self.toggle[0] = True

    def register_grow_line(self, data, nframes):
        self.line = self.ax[1].plot(np.zeros(2), np.ones(2)*data, '--', c='r')[0]
        self.static_dot = self.ax[1].plot(0, data, '*', c='r', ms=30, clip_on=False)[0]
        self.dline = 21./nframes
        self.toggle[1] = True

    def register_move_dot(self, data, sample_id, nframes):
        self.dot = self.ax[1].plot(0, data, '*', c='r', ms=30, clip_on=False)[0]
        canvas_xy = self.ax[0].transData.transform((sample_id,data))
        dest_xy = self.ax[1].transData.inverted().transform(canvas_xy)
        # self.ax[1].plot(dest_xy[0], dest_xy[1], 'o', c='b', clip_on=False)
        self.dxy = (dest_xy - [0, data])/nframes
        self.xmax = dest_xy[0]
        self.toggle[2] = True
    
    def step(self):
        # grow bars
        if self.toggle[0]:
            now_height = np.array([rect.get_height() for rect in self.bars])
            if np.any(now_height < self.hmax):
                new_height = now_height + self.dh
                for rect, h in zip(self.bars, new_height):
                    rect.set_height(h)
            else:
                self.toggle[0] = False
                self.play_next[1] = True
        # grow line
        if self.toggle[1]:
            xdata = self.line.get_xdata()
            if xdata[1] < 21:
                xdata[1] += self.dline
                self.line.set_xdata(xdata)
            else:
                self.toggle[1] = False
                self.play_next[2] = True
        # move dot
        if self.toggle[2]:
            data = self.dot.get_data()
            data = np.array((data[0],data[1])).flatten()
            if np.abs(data[0] - self.xmax)>1e-4:
                data += self.dxy
                self.dot.set_data(data)
            else:
                self.toggle[2] = False
                self.play_next[0] = True
                self.sample_set_counter += 1
                self.pause += 6

    def __call__(self, i):

        if i > 0 and self.sample_set_counter<10:
            if self.pause > 0:
                self.pause -= 1
                return self.bars
            if self.play_next[0]:
                self.reset_bar()
                self.register_grow_bars(self.data[self.sample_set_counter], self.grow_bars_duration)
                self.play_next[0] = False
                if self.line is not None:
                    self.line.remove()
                if self.static_dot is not None:
                    self.static_dot.remove()
            if self.play_next[1]:
                self.register_grow_line(self.data_mean[self.sample_set_counter], self.grow_line_duration)
                self.play_next[1] = False
            if self.play_next[2]:
                self.register_move_dot(self.data_mean[self.sample_set_counter], self.sample_set_counter+1, self.move_dot_duration)
                self.play_next[2] = False
            self.step()
        return self.bars
# data
np.random.seed(1901)
mean,std=605,np.sqrt(2283) #均值，标准差
size=(10, 20)
data=np.random.randn(*size)*std+mean
color=(data-data.min())/(data.max()-data.min())

# initialize figure
fig, ax = plt.subplots(2,1,figsize=(20,10), gridspec_kw=dict(
    left=0.10, right=0.98, top=0.87, bottom=0.15, hspace=0.1, height_ratios=[1, 2]))
x = np.arange(1,21)
ax[0].xaxis.tick_top()
ax[0].xaxis.set_label_position('top')
ax[0].spines['bottom'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[0].set_xlabel('采样组序号')
ax[0].set_ylabel('平均里程(km)', fontsize=35)
ax[0].set_xlim(0,11)
ax[0].set_xticks(np.arange(10)+1)
ax[0].set_ylim(570,630)
ax[0].set_yticks([570,600,630])
ax[1].set_xlim(0,21)
ax[1].set_ylim(450,740)
ax[1].set_ylabel('实际里程(km)', fontsize=35)
ax[1].set_xticks(np.arange(20)+1)
ax[1].set_xticklabels([])
ax[1].scatter(x, np.ones_like(x)*425, s=2500, marker=car_marker,clip_on=False, c='#353A71')
ud = UpdateFigures(ax, data)
plt.savefig(path/'test.pdf')
fps = 24
T = 14
anim = FuncAnimation(fig, ud, frames=fps*T+1, blit=True)
anim.save(path/'mean_est_demo.mp4', fps=fps, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

# %%
