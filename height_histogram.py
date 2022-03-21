# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from scipy.stats import norm
from matplotlib.animation import FuncAnimation
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 14
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

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
    person_marker = person_marker.transformed(Affine2D().rotate_deg(rotation))
    person_marker = person_marker.transformed(Affine2D().scale(-1,1))
    return person_marker

person_mkr = gen_marker('icons/person.svg',)

#%%
class UpdateFigure:
    def __init__(self, ax:plt.Axes, 
        ax_main:plt.Axes, ax_right:plt.Axes, ax_top:plt.Axes, ax_colorbar:int, data:np.ndarray):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of scatter plot
            ax_main (plt.Axes): axes of transfer function
            ax_right (plt.Axes): axes of histogram
            ax_top (plt.Axes): axes of histogram
            ax_colorbar (plt.Axes): axes of histogram
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
        self.transfer = lambda x: norm.ppf(x, loc=0.5, scale=0.15)
        self.transfer_grad_inv = lambda x: norm.pdf(x, loc=0.5, scale=0.15)
        self.transfer_inv = lambda y: norm.cdf(y, loc=0.5, scale=0.15)

        # ====================
        # generate the grid of person
        # ====================
        self.ax = ax
        xn, yn = 20, 20
        xx, yy = np.meshgrid(np.arange(xn), np.arange(yn))
        self.person_colors = np.ones((data.shape[0], 4))
        # self.person_colors = self.cm(self.transfer(data))
        self.persons = self.ax.scatter(
            xx.flatten(), yy.flatten(),
            s=1000, marker=person_mkr,
            facecolor=self.person_colors,
            )
        self.ax.invert_yaxis()
        # array to record the color of each flight
        # self.color = np.tile(self.colors['flight_init'],(int(n_days),1)).astype(float)
        # self.days = np.arange(data.shape[0])+1


        x_grid = np.linspace(0,1,100)
        # scatter plot:
        self.line_main, = ax_main.plot(x_grid, self.transfer(x_grid),
            color=self.colors['transfer'],lw=3)
        self.text = ax_main.text(0.5, 0.8, r'$y=F^{-1}(x)$', ha='left', 
            color=self.colors['transfer'], fontsize=26)
        ax_main.set_xlabel(r'$x$', fontsize=30)
        ax_main.set_ylabel(r'$y$', fontsize=30, rotation=0)

        # now determine nice limits by hand:
        ylim = (0, 1)
        xlim = (0, 1)
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        ax_main.set_xticks(xlim)
        ax_main.set_yticks(ylim)
        ax_main.set_xticklabels(xlim, color='w')
        ax_main.set_yticklabels(ylim, color='w')
        self.ax_main = ax_main
        # initialize the bins of histogram
        ax_top.set_ylabel(r'$f(x)$', fontsize=30, rotation=0)
        ax_right.set_xlabel(r'$f(y)$', fontsize=30)

        ax_top.set_xlim(self.ax_main.get_xlim())
        ax_top.set_ylim(0,40)
        ax_top.set_yticks([0,40])
        ax_top.set_yticklabels((0,40), color='w')
        ax_right.set_ylim(self.ax_main.get_ylim())
        ax_right.set_xlim(0,80)
        ax_right.set_xticks([])

        # fit the distribution with gaussian
        self.f1 = lambda x: np.ones_like(x)
        self.f2 = lambda x: self.f1(x)*self.transfer_grad_inv(self.transfer(x))
        self.fy = lambda x: self.f1(self.transfer_inv(x))*self.transfer_grad_inv(x)
            
        self.ax_top= ax_top
        self.ax_right= ax_right

        # ====================
        # histogram & bar plot
        # ====================

        bins = 20
        self.binsize=1.0/bins
        # self.x_counts, edges = np.histogram(data, range=(0,1), bins=bins,)
        self.x_counts = np.zeros(bins)
        edges = np.arange(0,1+self.binsize,self.binsize)
        self.bars_top = ax_top.bar(edges[:-1], height=self.x_counts, width=self.binsize, 
            align='edge', color=self.colors['f1'], alpha=0.7)

        # edge_center = edges[:-1]+self.binsize/2
        # self.transfer_index = (self.transfer(edge_center)//self.binsize).astype(int)
        # self.y_counts = [self.x_counts[self.transfer_index==i].sum() for i in range(self.x_counts.shape[0])]
        self.y_counts = np.zeros(bins)

        y_edges = self.transfer(edges)
        y_edges[0] = 0
        y_edges[-1] = 1
        self.y_inc = self.binsize/np.diff(y_edges)
        colors = self.cm(y_edges)
        self.bars_right = ax_right.barh(y_edges[:-1], 
            width=self.x_counts*self.y_inc, height=np.diff(y_edges), 
            align='edge', color=colors[1:], 
            alpha=0.7)
            
        # x_grid = np.linspace(0,1, 200)
        # factor = data.shape[0]*self.binsize
        # self.ax_top.plot(x_grid, self.f1(x_grid)*factor, color=self.colors['f1'], lw=5)
        # # self.ax_right.plot(self.f2(x_grid)*factor, self.transfer(x_grid), color=self.colors['f2'], lw=5)
        # colors = self.cm(self.transfer(np.linspace(0,1,200)))
        # for i in range(x_grid.shape[0]-1):
        #     self.ax_right.plot(self.f2(x_grid[i:i+2])*factor, self.transfer(x_grid[i:i+2]), color=colors[i+1], lw=5)
        
        # verts_t, verts_r = self.get_verts(xp)
        # self.shade_t = PolyCollection([verts_t], facecolor=self.colors['f1'], edgecolor='None', alpha=0.4) # Add a polygon instead of fill_between
        # self.shade_r = PolyCollection([verts_r], facecolor=self.colors['f2'], edgecolor='None', alpha=0.4) # Add a polygon instead of fill_between
        # ax_top.add_collection(self.shade_t)
        # ax_right.add_collection(self.shade_r)

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

        # ====================
        # draw colorbar
        # ====================
        gradient = np.atleast_2d(np.linspace(0, 1, 256))
        ax_colorbar.imshow(gradient, aspect='auto', cmap=self.cm, alpha=0.7)
        ax_colorbar.set_yticks([])
        ax_colorbar.set_xticks([0, 255])
        ax_colorbar.set_xticklabels(['低', '高'])
        ax_colorbar.set_title('身高', fontsize=20)


        # vmax = np.max((self.f1(x_grid).max(), self.f2(x_grid).max()))
        # vmin = np.min((self.f1(x_grid).min(), self.f2(x_grid).min()))

        # plt.colorbar(img, cax=ax_colorbar, orientation='horizontal')
        # ax_colorbar.set_xticks([vmin, vmax])
        # ax_colorbar.set_xticklabels(['low', 'high'])

        # self.text2 = ax.text(self.g_x(xp)*1.2,0.78,0, r"$g'(x)\Delta x$", ha='left', color='navy', fontsize=16)
        # self.text3 = ax.text(self.g_x(xp)*1.2,1,self.f_y(self.g_x(xp))*1.0, r"$f(g(x))$", ha='left', color='#FE8517', fontsize=16)
        self.xp= xp
        self.data = data
        self.last_hl_id = None

    @staticmethod
    def color_bar(bar, color_value, cmap=None, vmax=None, vmin=None):
        grad = np.atleast_2d(color_value)
        ax = bar.axes
        lim = ax.get_xlim()+ax.get_ylim()
        bar.set_zorder(1)
        bar.set_facecolor("none")
        x,y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        img = ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", zorder=0, cmap=cmap, vmax=vmax, vmin=vmin)
        ax.axis(lim)
        return img

    def get_verts(self, x_):
        x_array = np.arange(x_, x_+self.dxp+self.ddxp, self.ddxp)
        verts_t = [(xi, self.f1(xi)) for xi in x_array] \
                + [(x_+self.dxp, 0), (x_, 0)]

        verts_r = [(self.f2(xi), self.transfer(xi)) for xi in x_array] \
                + [(0, self.transfer(x_+self.dxp)), (0, self.transfer(x_))]
                
        return verts_t,verts_r

    def get_line_ends(self, xp):
        xp2 = np.ones(2)*xp
        return [
                [xp2, [self.transfer(xp), 1.2],],
                [xp2+self.dxp, [self.transfer(xp+self.dxp), 1.2],],
                [[xp, 1.2], self.transfer(xp2),],
                [[xp+self.dxp, 1.2], self.transfer(xp2+self.dxp),]
            ]

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i < self.data.shape[0]:

            if self.last_hl_id is not None:
                self.bars_top[self.last_hl_id].set_alpha(0.5)
                self.bars_right[self.last_hl_id].set_alpha(0.5)
                # self.bars_right[self.transfer_index[self.last_hl_id]].set_alpha(0.5)

            bin_id = int(self.data[i]//self.binsize)
            self.last_hl_id=bin_id
            # update the height of bars for histogram
            self.x_counts[bin_id] += 1
            self.bars_top[bin_id].set_height(self.x_counts[bin_id])
            self.bars_top[bin_id].set_alpha(1)
            self.y_counts[bin_id] += self.y_inc[bin_id]
            self.bars_right[bin_id].set_width(self.y_counts[bin_id])
            self.bars_right[bin_id].set_alpha(1)
            # self.y_counts[self.transfer_index[bin_id]] += 1
            # self.bars_right[self.transfer_index[bin_id]].set_width(self.y_counts[self.transfer_index[bin_id]])
            # self.bars_right[self.transfer_index[bin_id]].set_alpha(1)
            # update guiding lines
            xp = self.binsize*bin_id
            if xp == 0:
                xp += 0.01
            elif xp == 1:
                xp -= 0.01
            line_ends = self.get_line_ends(xp)
            [self.lines[i].set_data(*line_end) for i, line_end in enumerate(line_ends)]

            # update scatter facecolor
            self.person_colors[i] = self.cm(self.transfer(self.data[i]))
            self.persons.set_facecolor(self.person_colors)

        elif i == self.data.shape[0]:

            x_grid = np.linspace(0,1, 200)
            factor = self.data.shape[0]*self.binsize
            self.ax_top.plot(x_grid, self.f1(x_grid)*factor, color=self.colors['f1'], lw=5)
            # self.ax_right.plot(self.f2(x_grid)*factor, self.transfer(x_grid), color=self.colors['f2'], lw=5)
            colors = self.cm(self.transfer(np.linspace(0,1,200)))
            for i in range(x_grid.shape[0]-1):
                self.ax_right.plot(self.f2(x_grid[i:i+2])*factor, self.transfer(x_grid[i:i+2]), color=colors[i+1], lw=5)

        return self.bars_right

fig = plt.figure(figsize=(20,10),dpi=400)

ax = fig.add_gridspec(1, 1, left=0.05, right=0.5, top=0.95, bottom=0.05, figure=fig)
ax.axis('off')

# definitions for the axes
left, width = 0.55, 0.25
bottom, height = 0.15, 0.50
left_h = left + width + 0.01
bottom_h = bottom + height + 0.02
rect_main = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.15]
rect_histy = [left_h, bottom, 0.15, height]

rect_colorbar = [left_h+0.01, bottom_h+0.03, 0.15-0.02, 0.05]

axMain = plt.axes(rect_main)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)
axColorbar = plt.axes(rect_colorbar)

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
ud = UpdateFigure(ax, axMain, axHisty, axHistx, axColorbar, data)
nframes=450
plt.savefig('test.pdf')
# user FuncAnimation to generate frames of animation
# %%
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save('height_histogram.mp4', fps=12, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
