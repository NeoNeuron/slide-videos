# %%
from math import gamma
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from scipy.stats import norm
from matplotlib.animation import FuncAnimation
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 14
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

#%%
class UpdateFigure:
    def __init__(self, ax_main:plt.Axes, ax_right:plt.Axes, 
                 ax_top:plt.Axes, data:np.ndarray):
        """Plot the first frame for the animation.

        Args:
            ax_main (plt.Axes): axes of transfer function
            ax_right (plt.Axes): axes of histogram
            ax_top (plt.Axes): axes of histogram
            data (np.ndarray): random data for plotting
        """
        self.color_repo = dict(
            blue        = '#375492',
            green       = '#88E685',
            dark_green  = '#00683B',
            red         = '#93391E',
            pink        = '#E374B7',
            purple      = '#A268B4',
            black       = '#000000',
        )
        self.cm = plt.cm.RdYlBu_r
        self.colors = dict(
            mkr_init=[0,0,0,1],
            transfer=self.color_repo['blue'],
            f1      =self.color_repo['blue'],
            f2      =self.color_repo['green'],
            gl      =self.color_repo['black'],
        )
        # ====================
        # Define transfer functions
        # ====================
        # * define outside the inital
        # sigma=15
        # mu=800
        # a=-104
        # b=699
        # self.transfer = lambda x: a*np.log(norm.ppf(1-x)*sigma+mu)+b# if x>0.023577357735773578 else 0
        # self.transfer_inv_grad = lambda x: -norm.pdf((np.exp((x-b)/a)-mu)/sigma)*np.exp((x-b)/a)/a
        # self.transfer_inv = lambda y: 1-norm.cdf((np.exp((y-b)/a)-mu)/sigma)

        # ====================
        # generate the grid of person
        # ====================
        x_grid = np.linspace(0,1,1000)
        # scatter plot:
        self.line_main, = ax_main.plot(x_grid, self.transfer(x_grid),
            color=self.colors['transfer'],lw=3)
        self.text = ax_main.text(0.45, 0.5, r'$x=F_x^{-1}(y)$', ha='left', 
            color=self.colors['transfer'], transform=ax_main.transAxes, fontsize=40)

        # now determine nice limits by hand:
        ylim = (0, 12)
        xlim = (0, 1)
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        self.ax_main = ax_main
        self.ax_main.grid()
        # initialize the bins of histogram
        ax_main.text(0.5, -0.12,   r'$y$',    transform=ax_main.transAxes,  fontsize=30)
        ax_main.text(-0.10, 0.5, r'$x$',    transform=ax_main.transAxes,  fontsize=30)
        ax_top.text(-0.160, 0.55,  r'$f(y)$', transform=ax_top.transAxes,   fontsize=30)
        ax_right.text(0.3, -0.12,  r'$f(x)$', transform=ax_right.transAxes, fontsize=30)

        ax_top.set_xlim(self.ax_main.get_xlim())
        ax_top.set_ylim(0,2)
        ax_top.set_yticks([])
        ax_right.set_ylim(self.ax_main.get_ylim())
        ax_right.set_xlim(0,0.3)
        ax_right.set_xticks([])

        # fit the distribution with gaussian
        self.f1 = lambda x: np.ones_like(x)
        self.f2 = lambda x: self.f1(x)*self.transfer_inv_grad(self.transfer(x))
            
        self.ax_top= ax_top
        self.ax_right= ax_right

        # ====================
        # histogram & bar plot
        # ====================

        bins = 40
        self.binsize=1.0/bins
        # self.x_counts, edges = np.histogram(data, range=(0,1), bins=bins,)
        self.x_counts = np.zeros(bins)
        self.x_counts_density = self.x_counts/data.shape[0]/self.binsize
        edges = np.arange(0,1+self.binsize,self.binsize)
        self.bars_top = ax_top.bar(edges[:-1], height=self.x_counts_density, width=self.binsize, 
            align='edge', color=self.colors['f1'], alpha=0.7)

        self.y_counts = np.zeros(bins)

        y_edges = self.transfer(edges)
        y_edges[-1] = 12
        y_edges[0] = -1
        self.y_inc = 1.0/data.shape[0]/np.diff(y_edges)
        self.bars_right = ax_right.barh(y_edges[:-1], 
            width=self.x_counts*self.y_inc, height=np.diff(y_edges), 
            align='edge', color=self.colors['f2'], 
            alpha=0.7)

        # ====================
        # draw points
        # ====================
        xp = 0.3
        self.ddxp = 0.0001
        self.dxp = self.binsize

        # ====================
        # draw guiding lines
        # ====================

        line_ends = self.get_line_ends(xp)

        self.lines = [
            ax_main.plot(*line_end, ls='--', color=self.colors['gl'], alpha=1)[0]
            for line_end in line_ends
            ]
        [line.set_clip_on(False) for line in self.lines]

        self.xp= xp
        self.data = data
        self.last_hl_id = None

        # self.ax_top.plot(x_grid, self.f1(x_grid), color=self.colors['f1'], lw=5)
        # self.ax_right.plot(self.f2(x_grid)*0.068259, self.transfer(x_grid), color=self.colors['f2'], lw=5)
        # x = np.linspace(0, 16,1000)
        # y = np.array([np.exp(-4)*4**xi/gamma(xi+1) for xi in x])*100
        # ax_right.plot(y,x,c='b')

    @staticmethod
    def transfer(x):
        sigma=15
        mu=800
        a=-104
        b=699
        if isinstance(x, np.ndarray):
            y = a*np.log(norm.ppf(1-x)*sigma+mu)+b
            mask = x<=0.023577357735773578
            y[mask] = 0
            return y
        else:
            if x>0.023577357735773578:
                return a*np.log(norm.ppf(1-x)*sigma+mu)+b
            else:
                return 0

    @staticmethod
    def transfer_inv_grad(x):
        sigma=15
        mu=800
        a=-104
        b=699
        if isinstance(x, np.ndarray):
            y = -norm.pdf((np.exp((x-b)/a)-mu)/sigma)*np.exp((x-b)/a)/a
            mask = x<0
            y[mask]=0
            return y
        else:
            if x>=0:
                return -norm.pdf((np.exp((x-b)/a)-mu)/sigma)*np.exp((x-b)/a)/a
            else:
                return 0

    @staticmethod
    def transfer_inv(y):
        sigma=15
        mu=800
        a=-104
        b=699
        if isinstance(y, np.ndarray):
            x = 1-norm.cdf((np.exp((y-b)/a)-mu)/sigma)
            mask = y<0
            x[mask]=0
            return x
        else:
            if y>=0:
                return 1-norm.cdf((np.exp((y-b)/a)-mu)/sigma)
            else:
                return 0

    def get_line_ends(self, xp):
        xp2 = np.ones(2)*xp
        return [
                [xp2, [self.transfer(xp), 13],],
                [xp2+self.dxp, [self.transfer(xp+self.dxp), 13],],
                [[xp, 1.02], self.transfer(xp2),],
                [[xp+self.dxp, 1.02], self.transfer(xp2+self.dxp),]
            ]

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i>0 and i < self.data.shape[0]:

            if self.last_hl_id is not None:
                self.bars_top[self.last_hl_id].set_alpha(0.5)
                self.bars_right[self.last_hl_id].set_alpha(0.5)
                # self.bars_right[self.transfer_index[self.last_hl_id]].set_alpha(0.5)

            bin_id = int(self.data[i-1]//self.binsize)
            self.last_hl_id=bin_id
            # update the height of bars for histogram
            self.x_counts[bin_id] += 1.0/self.data.shape[0]/self.binsize
            self.bars_top[bin_id].set_height(self.x_counts[bin_id])
            self.bars_top[bin_id].set_alpha(1)
            self.y_counts[bin_id] += self.y_inc[bin_id]
            self.bars_right[bin_id].set_width(self.y_counts[bin_id])
            self.bars_right[bin_id].set_alpha(1)
            # update guiding lines
            xp = self.binsize*bin_id
            if xp == 0:
                xp += 0.001
            elif xp == 1:
                xp -= 0.001
            line_ends = self.get_line_ends(xp)
            [self.lines[i].set_data(*line_end) for i, line_end in enumerate(line_ends)]

        elif i > self.data.shape[0]:

            x_grid = np.linspace(0,1, 200)
            self.ax_top.plot(x_grid, self.f1(x_grid), color=self.colors['f1'], lw=5)
            self.ax_right.plot(self.f2(x_grid)*0.068259, self.transfer(x_grid), color=self.colors['f2'], lw=5)

        return self.bars_right

fig = plt.figure(figsize=(12,10),dpi=400)

# definitions for the axes
left, width = 0.10, 0.6
bottom, height = 0.10, 0.6
left_h = left + width + 0.01
bottom_h = bottom + height + 0.02
rect_main = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.15]
rect_histy = [left_h, bottom, 0.15, height]

axMain = plt.axes(rect_main)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

axHistx.spines["top"].set_visible(False)
axHistx.spines["right"].set_visible(False)
axHistx.spines["left"].set_visible(False)
axHisty.spines["top"].set_visible(False)
axHisty.spines["right"].set_visible(False)
axHisty.spines["bottom"].set_visible(False)

# no labels
from matplotlib.ticker import NullFormatter 
nullfmt = NullFormatter()         # no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# create a figure updater
np.random.seed(0)
data = np.random.rand(400)
ud = UpdateFigure(axMain, axHisty, axHistx, data)
nframes=432
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save('random_gen_demo.mp4', fps=12, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
#%%
from scipy import stats
fig = plt. figure(figsize=(20,10), dpi=200)
xticks=[0,2,4,6,8,10,12,14]
yticks=[0,500,1000,1500,2000]
yticklabels=[0,0.05,0.1,0.15,0.2]
size = 10000
a=stats.norm.rvs(800, 15, size=size, random_state=2)
b = np.log(a)
c = -(b - min(b)) * (15 / (max(b) - min(b))) + 11
print('a = ',-15 / (max(b) - min(b)))
print('b = ', 15*min(b) / (max (b) - min(b))+ 11)
d = plt.hist(c, bins = 15, range=[0, 15], color='b', alpha=0.3)

x = np.linspace(0, 16,1000)
y = np.array([np.exp(-4)*4**xi/gamma(xi+1) for xi in x])*10000
plt.plot(x, y)
plt.grid()
plt.xticks(xticks)
plt.yticks(yticks,yticklabels)
# %%
x_grid = np.linspace(0,16,1000)
sigma=15
mu=800
a=-104
b=699
transfer_inv_grad = lambda x: -norm.pdf((np.exp((x-b)/a)-mu)/sigma)*np.exp((x-b)/a)/a
plt.plot(x_grid, transfer_inv_grad(x_grid), label='fitting')
y = np.array([np.exp(-4)*4**xi/gamma(xi+1) for xi in x])*15
plt.plot(x,y,c='r', label='target')
plt.legend()