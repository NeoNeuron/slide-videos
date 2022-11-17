# %%
from pathlib import Path
path = Path('./function_of_random_variables/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 14
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

#%%
class UpdateFigure:
    def __init__(self, 
        ax_main:plt.Axes, ax_right:plt.Axes, ax_top:plt.Axes, ax_colorbar:int):
        """Plot the first frame for the animation.

        Args:
            ax_main (plt.Axes): axes of scatter plot
            ax_right (plt.Axes): axes of histogram
            ax_top (plt.Axes): axes of line plot
            n_days (int): number of days to plot
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
        self.cmap = 'GnBu_r'
        self.colors = dict(
            transfer=self.color_repo['blue'],
            f1      =self.color_repo['pink'],
            f2      =self.color_repo['green'],
            gl = self.color_repo['black'],
        )
        # generate the grid of flights
        self.transfer = lambda x: np.log(x+1)/np.log(3)
        self.transfer_grad = lambda x: 1/(x+1)/np.log(3)
        self.transfer_inv = lambda y: np.exp(y*np.log(3))-1

        x_grid = np.linspace(0,2,100)
        # scatter plot:
        self.line_main, = ax_main.plot(x_grid, self.transfer(x_grid),lw=3,color=self.colors['transfer'],)
        self.text = ax_main.text(1.2, 0.75, r'$y=g(x)$', ha='right', color=self.colors['transfer'], fontsize=26)
        ax_main.set_xlabel(r'$x$', fontsize=30, labelpad=-15)
        ax_main.set_ylabel(r'$y$', fontsize=30, rotation=0)

        # now determine nice limits by hand:
        ylim = (0, 1)
        xlim = (0, 2)
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        ax_main.set_xticks(xlim)
        ax_main.set_yticks(ylim)
        ax_main.set_xticklabels(xlim, color='w')
        ax_main.set_yticklabels(ylim, color='w')
        self.ax_main = ax_main
        # initialize the bins of histogram
        ax_top.set_ylabel(r'$f(x)$', fontsize=30, rotation=0)
        ax_right.set_xlabel(r'$f(y)$', fontsize=30, labelpad=-10)

        ax_top.set_xlim(self.ax_main.get_xlim())
        ax_top.set_ylim(0,2.1)
        ax_top.set_yticks([])
        ax_right.set_ylim(self.ax_main.get_ylim())
        ax_right.set_xlim(0,4.2)
        ax_right.set_xticks([])

        # fit the distribution with gaussian
        self.f1 = lambda x: norm.pdf(x, loc=1, scale=0.25)
        self.f2 = lambda x: self.f1(x)/self.transfer_grad(x)
        self.fy = lambda x: self.f1(self.transfer_inv(x))/self.transfer_grad(self.transfer_inv(x))

        self.stick_width = 0.4
        x_grid = np.linspace(*xlim, 200)
        ax_top.plot(x_grid, self.f1(x_grid)+self.stick_width, color=self.colors['f1'], lw=5)
        ax_right.plot(self.f2(x_grid)+self.stick_width, self.transfer(x_grid), color=self.colors['f2'], lw=5)

        self.ax_top= ax_top
        self.ax_right= ax_right

        # ====================
        # draw points
        # ====================
        xp = 0.3
        self.ddxp = 1.5e-4
        self.dxp = 0.1

        # ====================
        # shading areas
        # ====================
        verts_t, verts_r = self.get_verts(xp)
        self.shade_t = PolyCollection([verts_t], facecolor=self.colors['f1'], edgecolor='None', alpha=0.4) # Add a polygon instead of fill_between
        self.shade_r = PolyCollection([verts_r], facecolor=self.colors['f2'], edgecolor='None', alpha=0.4) # Add a polygon instead of fill_between
        ax_top.add_collection(self.shade_t)
        ax_right.add_collection(self.shade_r)

        # ====================
        # draw guiding lines
        # ====================

        line_ends = self.get_line_ends(xp)

        self.lines = [
            ax_main.plot(*line_end, ls='--', color=self.colors['gl'], alpha=0.5)[0]
            for line_end in line_ends
            ]
        [line.set_clip_on(False) for line in self.lines]

        # ====================
        # draw sticks
        # ====================
        y_grid = np.linspace(0,1,100)
        self.barh, = ax_top.barh(y=0, width=2, height=self.stick_width, align='edge')
        self.barv, = ax_right.bar(x=0, width=self.stick_width, height=1, align='edge')
        vmax = np.max((self.f1(x_grid).max(), self.f2(x_grid).max()))
        vmin = np.min((self.f1(x_grid).min(), self.f2(x_grid).min()))
        img = self.color_bar(self.barh, self.f1(x_grid), vmax=vmax, vmin=vmin, cmap=self.cmap)
        self.color_bar(self.barv, np.atleast_2d(np.flip(self.fy(y_grid))).T, vmax=vmax, vmin=vmin, cmap=self.cmap)
        plt.colorbar(img, cax=ax_colorbar, orientation='horizontal')
        ax_colorbar.set_title('密度', fontsize=20)
        ax_colorbar.set_xticks([vmin, vmax])
        ax_colorbar.set_xticklabels(['低', '高'])

        # self.text2 = ax.text(self.g_x(xp)*1.2,0.78,0, r"$g'(x)\Delta x$", ha='left', color='navy', fontsize=16)
        # self.text3 = ax.text(self.g_x(xp)*1.2,1,self.f_y(self.g_x(xp))*1.0, r"$f(g(x))$", ha='left', color='#FE8517', fontsize=16)
        self.xp= xp

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
        verts_t = [(xi, self.f1(xi)+self.stick_width) for xi in x_array] \
                + [(x_+self.dxp, 0), (x_, 0)]

        verts_r = [(self.f2(xi)+self.stick_width, self.transfer(xi)) for xi in x_array] \
                + [(0, self.transfer(x_+self.dxp)), (0, self.transfer(x_))]
                
        return verts_t,verts_r

    def get_line_ends(self, xp):
        xp2 = np.ones(2)*xp
        return [
                [xp2, [self.transfer(xp), 1.2],],
                [xp2+self.dxp, [self.transfer(xp+self.dxp), 1.2],],
                [[xp, 2.2], self.transfer(xp2),],
                [[xp+self.dxp, 2.2], self.transfer(xp2+self.dxp),]
            ]

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            self.xp += self.ddxp*50
            verts_t, verts_r = self.get_verts(self.xp)
            self.shade_t.set_verts([verts_t])
            self.shade_r.set_verts([verts_r])

            line_ends = self.get_line_ends(self.xp)
            [self.lines[i].set_data(*line_end) for i, line_end in enumerate(line_ends)]

        return [self.shade_t,]

fig = plt.figure(figsize=(10,5))
# definitions for the axes
left, width = 0.1, 0.55
bottom, height = 0.1, 0.55
left_h = left + width + 0.02
bottom_h = bottom + height + 0.02
rect_main = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.3]
rect_histy = [left_h, bottom, 0.3, height]

rect_colorbar = [left_h+0.03, bottom_h+0.08, 0.25, 0.05]

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
ud = UpdateFigure(axMain, axHisty, axHistx, axColorbar)
plt.savefig(path/'test_stick.pdf')
# user FuncAnimation to generate frames of animation
# %%
nframes=48*3
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save(path/'stick_density.mp4', fps=48, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%