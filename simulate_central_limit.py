#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 16
from matplotlib.animation import FuncAnimation
from scipy.stats import binom, bernoulli, boltzmann, norm

class UpdateHistogram():
    '''Update constrained historgram.
    
    Bin size of histogram is 1.
    '''
    def __init__(self, ax, data, range=(0,350), 
        zscore = False, envelope_curve=False,
        autolim=True, fade = False,
        xlabel_scale=False):

        self.ax = ax
        self.data = data
        self.zscore = zscore
        self.envelope_curve=envelope_curve
        self.bar_width=1
        self.xlabel_scale = xlabel_scale
        if range is None:
            raise RuntimeError('Missing range parameter.')
        self.bins = int((range[1]-range[0] + 1)/self.bar_width)
        self.range = [range[0]-0.5, range[1]+0.5]
        self.edges = np.linspace(*self.range, self.bins+1)
        self.edge_centers = (self.edges[1:]+self.edges[:-1])/2
        self.rects = self.ax.bar(
            self.edge_centers, np.zeros(self.bins),
            width = self.bar_width,
            color='#108B96')
        # gauss = norm(loc=mean, scale=var)
        # self.ax.plot(edges, gauss(edges), ls='--', color='#B72C31', label='高斯模型估计')
        # self.ax.set_ylim(0,gauss(edges).max()*1.2)
        # self.ax.legend(loc=1, fontsize=14)
        # self.ax.set_xlim(*range)
        if autolim:
            if autolim == 'x':
                self.autoxlim = True
                self.autoylim = False
            elif autolim == 'y':
                self.autoxlim = False
                self.autoylim = True
            else:
                self.autoxlim=self.autoylim=True
        else:
            self.autoxlim=self.autoylim=False
        self.ax.set_ylim(0,1.1)
        if self.zscore:
            self.ax.set_xlim(-10,10)
        else:
            self.ax.set_xlim(-0.5,26)
        if self.xlabel_scale:
            self.ax.set_xticklabels(self.ax.get_xticks()*self.xlabel_scale)
        self.ax.set_ylabel('概率密度')
        self.ax.set_xlabel('需要登机人数')
        self.ax.set_title(f'售票数 : {0:5d}', fontsize=20)
        self.number_of_sample_list = [1,2,3,4,5,8,12,18,28,43,65,99,151,230, 350]
        self.color = plt.cm.Oranges(0.8*np.arange(len(self.number_of_sample_list))/len(self.number_of_sample_list))
        self.alpha = np.linspace(0.1, 0.8, self.color.shape[0])
        self.alpha[-1]=1
        self.fade = fade
        self.lines = []

    def set_colors(self, color_set):
        self.color = color_set
        self.alpha = np.linspace(0.1, 0.8, self.color.shape[0])
        self.alpha[-1]=1

    def set_frame_numbers(self, number_set):
        self.number_of_sample_list = number_set
    

    def __call__(self, i):
        if i>= len(self.number_of_sample_list):
            idx = self.number_of_sample_list[-1]
        else:
            idx = self.number_of_sample_list[i]
        margin = np.sum(self.data[:,:idx], axis=1, dtype=float)
        bins = self.bins
        if self.zscore:
            scaling = margin.std()
            # scaling = idx
            range = ((val-margin.mean())/scaling for val in self.range)
            margin = (margin - margin.mean())/scaling
        else:
            range = self.range

        counts, edges = np.histogram(margin, bins=bins, range=range, density=True)
        # counts /= counts.max()
        width = edges[1]-edges[0]
        for rect, h, x in zip(self.rects, counts, edges):
            rect.set_x(x)
            rect.set_height(h)
            rect.set_width(width)
        if i >= 0 and i < len(self.number_of_sample_list) and self.envelope_curve:
            if self.envelope_curve == 'gauss':
                self._draw_gauss(i)
            elif self.envelope_curve == 'joint':
                self._draw_conti_hist(counts, edges, i)
            else:
                raise ValueError("Invalid argument for envelope_curve.")
            self._recolor()

        # adjust xlim
        if self.autoxlim:
            last_nonzero_pos = self.edge_centers[counts > 0][-1]
            first_nonzero_pos = self.edge_centers[counts > 0][0]
            xlim = self.ax.get_xlim()
            if xlim[0]>=first_nonzero_pos:
                self.ax.set_xlim(-xlim[1], xlim[1])
                if self.xlabel_scale:
                    self.ax.set_xticklabels(self.ax.get_xticks()*self.xlabel_scale)
            if xlim[1]<=last_nonzero_pos:
                self.ax.set_xlim(xlim[0], xlim[1]*2)
                if self.xlabel_scale:
                    self.ax.set_xticklabels(self.ax.get_xticks()*self.xlabel_scale)

        # adjust ylim
        if self.autoylim:
            ylim = self.ax.get_ylim()
            if ylim[1]>=10*counts.max():
                self.ax.set_ylim(ylim[0], ylim[1]/5)
        
        # dark red : '#B72C31'
        self.ax.set_title(self.ax.get_title()[:-5]+f'{idx:5d}', fontsize=20)
        return self.rects

    def _draw_gauss(self, i):
        margin = np.sum(self.data[:,:self.number_of_sample_list[i]], axis=1, dtype=float)
        gauss = norm(loc=margin.mean(), scale=margin.std())
        xlim = self.ax.get_xlim()
        x_gauss = np.linspace(xlim[0], xlim[-1], 101)
        line, = self.ax.plot(
            x_gauss, gauss.pdf(x_gauss), 
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
        for line, c, alpha in zip(self.lines, self.color[-len(self.lines):], self.alpha[-len(self.lines):]):
            line.set_color(c)
            if self.fade:
                line.set_alpha(alpha)

class UpdateCurve():
    '''Update curve.
    
    '''
    def __init__(self, ax, data,
        autolim=True, xlabel_scale=False):

        self.ax = ax
        self.data = data
        self.n_grid = np.arange(self.data.shape[0])
        self.bar_width=1
        self.xlabel_scale = xlabel_scale
        self.rects = self.ax.bar([],[])
        self.line, = self.ax.plot([],[],'-o', ms=3)
        if autolim:
            if autolim == 'x':
                self.autoxlim = True
                self.autoylim = False
            elif autolim == 'y':
                self.autoxlim = False
                self.autoylim = True
            else:
                self.autoxlim=self.autoylim=True
        else:
            self.autoxlim=self.autoylim=False
        self.ax.set_ylim(data.min(),data.max())
        self.ax.set_xlim(-0.5,26)
        if self.xlabel_scale:
            self.ax.set_xticklabels(self.ax.get_xticks()*self.xlabel_scale)
        self.ax.set_ylabel('算术平均值')
        self.ax.set_xlabel('样本数 n')
        self.ax.set_title(f'n = : {0:5d}', fontsize=20)
        self.number_of_sample_list = np.array([1,2,3,4,5,8,12,18,28,43,65,99,151,230, 350])

    def __call__(self, i):
        if i>= len(self.number_of_sample_list):
            idx = self.number_of_sample_list[-1]
        else:
            idx = self.number_of_sample_list[i]
        self.line.set_data(self.n_grid[:idx], self.data[:idx])

        # adjust xlim
        if self.autoxlim:
            if idx<self.n_grid.shape[0]:
                last_nonzero_pos = self.n_grid[idx]
                xlim = self.ax.get_xlim()
                # if xlim[0]>=first_nonzero_pos:
                #     self.ax.set_xlim(-xlim[1], xlim[1])
                #     if self.xlabel_scale:
                #         self.ax.set_xticklabels(self.ax.get_xticks()*self.xlabel_scale)
                if xlim[1]<=last_nonzero_pos:
                    self.ax.set_xlim(xlim[0], xlim[1]*2)
                    if self.xlabel_scale:
                        self.ax.set_xticklabels(self.ax.get_xticks()*self.xlabel_scale)

        # # adjust ylim
        # if self.autoylim:
        #     ylim = self.ax.get_ylim()
        #     if ylim[1]>=10*self.data[:idx].max():
        #         self.ax.set_ylim(ylim[0], ylim[1]/5)
        
        self.ax.set_title(self.ax.get_title()[:-5]+f'{idx:5d}', fontsize=20)
        return self.rects

    def set_frame_numbers(self, number_set):
        self.number_of_sample_list = number_set
#%%
if __name__ == '__main__':
    # Simulate Bernoulli random tests
    #%%
    from pathlib import Path
    path = Path('./central_limit_theorem/')
    path.mkdir(exist_ok=True)
    my_distribution = bernoulli
    my_dist_args = dict(
        p=0.9,
    )

    n = 350 # number of accumulated samples
    K = 100000 # number of random tests
    # generate sampling data
    attendence = my_distribution.rvs(**my_dist_args, size=(K,n), random_state=1240)

    fig, ax = plt.subplots(
        1,1, figsize=(4,3),
        gridspec_kw=dict(left=0.18, right=0.95, bottom=0.18))

    zscore = False
    uh = UpdateHistogram(
        ax, attendence, (-n,n), 
        zscore=zscore, autolim=not zscore, 
        fade=False, envelope_curve='joint')
    uh.ax.set_ylim(0,0.5)
    # if zscore:
    #     uh.ax.set_xlabel(r'$\frac{Y_n-EY_n}{\sqrt{DY_n}}$', fontsize=14)
    #     x_grid = np.linspace(-10,10,400)
    #     normal_curve = Gaussian(0,1)(x_grid)/(x_grid[1]-x_grid[0])
    #     uh.ax.plot(x_grid, normal_curve, 'r')
    # else:
    #     uh.ax.set_xlabel(r'$Y_n$', fontsize=14)
    # uh.ax.set_title(r'$n$ : '+uh.ax.get_title()[-5:], fontsize=20)
    number_list = np.array([1,2,3,4,5,6,8,12,18,28,43,65,99,151,230,350])
    number_list = np.tile(number_list, (3,1)).T.flatten()
    uh.set_frame_numbers(number_list)
    uh.set_colors(plt.cm.Oranges(0.8*np.arange(len(number_list))/len(number_list)))

    anim = FuncAnimation(fig, uh, frames=16*3, blit=True)
    fname = "evolving_bernoulli.mp4"
    anim.save(path/fname, fps=4, dpi=300, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

    # %%

    fig, ax = plt.subplots(
        1,1, figsize=(4,3),
        gridspec_kw=dict(left=0.18, right=0.95, bottom=0.18))
    uh = UpdateHistogram(
        ax, attendence, (-n,n), 
        zscore=False, autolim=False, 
        fade=False, envelope_curve='gauss')
    number_list = np.array([1,2,3,4,5,6,8,12,18,28,43,65,99,151,230,350])
    uh.set_frame_numbers(number_list)
    colors = np.zeros((len(number_list), 3))
    colors[:,0] = 1
    uh.ax.grid()
    uh.set_colors(colors)
    uh.ax.set_xlim(-1,3)
    uh.ax.set_ylim(0,1.4)
    uh(0)
    fig.savefig(path/'frame0.png', dpi=300)
    uh.ax.set_xlim(5,15)
    uh.ax.set_ylim(0,0.5)
    uh(7)
    fig.savefig(path/'frame7.png', dpi=300)
    uh.ax.set_xlim(280,360)
    uh.ax.set_ylim(0,0.09)
    uh(15)
    fig.savefig(path/'frame15.png', dpi=300)
# %%
