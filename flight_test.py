# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.stats import bernoulli
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24

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

flight_marker = gen_marker('icons/flight.svg',0)
#%%
# test marker
xx, yy = np.meshgrid(np.arange(4), np.arange(4))
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.scatter(xx.flatten(),yy.flatten(),  s=2000, color='k', marker=flight_marker )
ax.axis('off')
#%%
class UpdateFigure:
    def __init__(self, 
        ax:plt.Axes, data:np.ndarray, ax_main:plt.Axes, 
        ax_right:plt.Axes, ax_top:plt.Axes, n_days:int):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
            data (np.ndarray): 1-D array of number of passagers for each days
            ax_main (plt.Axes): axes of scatter plot
            ax_right (plt.Axes): axes of histogram
            ax_top (plt.Axes): axes of line plot
            n_days (int): number of days to plot
        """

        self.colors = dict(
            flight_init=[0,0,0,1],
            main=np.array([0,109,135,255])/255.0, #006D87
            gauss=np.array([177,90,67,255])/255.0, #B15A43
            flight_red=np.array([230,0,18,255])/255.0,
            flight_green=np.array([0,176,80,255])/255.0,
        )
        self.ax = ax
        # generate the grid of flights
        xn, yn = 37, 10
        xx, yy = np.meshgrid(np.arange(xn), np.arange(yn))
        self.sc_flight = self.ax.scatter(
            xx.flatten()[:n_days], yy.flatten()[:n_days], s=2000, 
            facecolor=self.colors['flight_init'], marker=flight_marker)
        # array to record the color of each flight
        self.color = np.tile(self.colors['flight_init'],(int(n_days),1)).astype(float)
        self.ax.set_xlim(-1,xn)
        self.ax.set_ylim(-1,yn)
        self.ax.invert_yaxis()
        self.data = data
        self.days = np.arange(data.shape[0])+1

        # scatter plot:
        self.line_main, = ax_main.plot([], [], 'o',
            color=self.colors['main'],
            markersize=6,
            markerfacecolor='none',
            markeredgewidth=2)
        ax_main.set_xlabel('日期', fontsize=40)
        ax_main.set_ylabel('登机人数', fontsize=40)

        # now determine nice limits by hand:
        ymax = np.max(np.fabs(data))
        ymin = np.min(np.fabs(data))
        xlim = (-1, data.shape[0]+1)
        ylim = (int(ymin*0.95), int(ymax*1.05))
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        self.ax_main = ax_main

        # initialize the bins of histogram
        self.bins = int(ylim[1]-ylim[0] + 1)
        self.range = [ylim[0]-0.5, ylim[1]+0.5]
        counts, edges = np.histogram(data, range=self.range, bins = self.bins)
        self.rects = ax_right.barh((edges[1:]+edges[:-1])/2, np.zeros_like(counts), color=self.colors['main'])
        self.data_line_top = (np.cumsum(data<=300)/(np.arange(data.shape[0])+1))*100
        self.line_top, = ax_top.plot([],[], color=self.colors['main'], lw=5)
        ax_top.set_ylabel('乘客均有座概率(%)', fontsize=40)
        ax_right.set_xlabel('天数', fontsize=40)

        ax_top.set_xlim(self.ax_main.get_xlim())
        ax_top.set_ylim(96.9,100.1)
        ax_right.set_ylim(self.ax_main.get_ylim())
        ax_right.set_xlim(0,36)

        # fit the distribution with gaussian
        func = lambda x,b,c: np.exp(-(x-b)**2/2/c)
            
        y_grid = np.linspace(ylim[0], ylim[1], 200)
        gauss_curve = func(y_grid, 0.9*seats, 0.09*seats)
        gauss_curve /= np.sum(gauss_curve*(y_grid[1]-y_grid[0]))
        gauss_curve *= n_days
        ax_right.plot(gauss_curve, y_grid, color=self.colors['gaussian'], lw=5)

        self.ax_top= ax_top
        self.ax_right= ax_right

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            n_inc = 1
            # update lines
            idx = (i)*n_inc
            self.line_main.set_data(self.days[:idx], self.data[:idx])
            self.line_top.set_data(self.days[:idx], self.data_line_top[:idx])
            
            # update the height of bars for histogram
            counts, edges = np.histogram(self.data[:idx], range=self.range, bins = self.bins)
            for rect, h in zip(self.rects, counts):
                rect.set_width(h)
            # update scatter facecolor
            for j in np.arange(n_inc):
                idx = (i-1)*n_inc+j
                self.color[idx,:] = self.colors['flight_red'] if self.data[idx]>300 else self.colors['flight_green']
            self.sc_flight.set_facecolor(self.color)
        return self.rects
# %%
my_distribution = bernoulli
my_dist_args = dict(
    p=0.9,
)

n = 350 # number of accumulated samples
K = 100000 # number of random tests

# calculate the accumulate mean and variance
single_mean, single_var  = my_distribution.stats(**my_dist_args, moments='mv')
# generate sampling data
attendence = my_distribution.rvs(**my_dist_args, size=(K,n), random_state=99239)
days = 365
seats = 319
margin = np.sum(attendence[100:100+days,:seats], axis=1, dtype=float)

fig = plt.figure(figsize=(30,20),dpi=100)
spec1 = gridspec.GridSpec(ncols=1, nrows=1, left=0.04, right=0.96, top=0.98, bottom=0.60, figure=fig)
ax0 = fig.add_subplot(spec1[0])
ax0.axis('off')

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.27
left_h = left + width + 0.02
bottom_h = bottom + height + 0.02
rect_main = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.17, height]

axMain = plt.axes(rect_main)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
from matplotlib.ticker import NullFormatter 
nullfmt = NullFormatter()         # no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# create a figure updater
ud = UpdateFigure(ax0, margin, axMain, axHisty, axHistx, days)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=366, blit=True)
# save animation as *.mp4
anim.save('flight_movie.mp4', fps=24, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%