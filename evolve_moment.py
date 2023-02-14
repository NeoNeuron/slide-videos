# %%
from pathlib import Path
path = Path('./moment_estimation/')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.animation import FuncAnimation
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
#%%
class UpdateFigure:
    def __init__(self, 
        ax:plt.Axes, x:np.ndarray, var_name:str, var_range:tuple, color, text_pos:tuple):
        """

        Args:
            ax (plt.Axes): _description_
            x (np.ndarray): _description_
            var_name (str): _description_
            var_range (tuple): _description_
            color (_type_): _description_
            text_pos (tuple): _description_
        """

        self.ax = ax
        self.x = x
        self.mean = var_range[0] if var_name=='mean' else 0.0
        self.std = var_range[0] if var_name=='std' else 1.0
        self.var_name = var_name
        self.var_range = var_range
        self.dx = (var_range[1]-var_range[0])/100.0
        self.c=color
        self.text_pos=text_pos

        y = norm.pdf(self.x, loc=self.mean, scale=self.std)
        self.line = self.ax.plot(self.x, y, color=self.c, zorder=0)

    def update_gauss(self, y):
        self.line[0].set_data(self.x,y)

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i < 101:
            if self.var_name == 'mean':
                y = norm.pdf(self.x,loc=self.mean+i*self.dx, scale=self.std)
            elif self.var_name == 'std':
                y = norm.pdf(self.x,loc=self.mean, scale=self.std+i*self.dx)
            self.update_gauss(y)
        elif i == 101:
            if self.var_name == 'mean':
                text = r'$EX=$%2d'%self.var_range[1]
            elif self.var_name == 'std':
                text = r'$EX^2=$%2d'%self.var_range[1]
            self.ax.text(self.text_pos[0], self.text_pos[1], 
                         text, ha='left',usetex=True,
                         fontsize=25, color=self.c, 
                         transform=self.ax.transAxes)
        else:
            pass
        return self.line

#! ============================================================
#! evolve std
fig, ax = plt.subplots(1,1, figsize=(5,3.5), dpi=400)
x = np.linspace(-6, 6, 401)
y = norm.pdf(x, loc=0, scale=1)
ax.plot(x, y, color='b', zorder=1)
ax.text(0.75, 0.88, 
        r'$EX^2=$%2d'%(1.0), ha='left',usetex=True,
        fontsize=25, color='b', 
        transform=ax.transAxes)
ax.set_xlim(-6.0, 6.0)
ax.set_xticks([-6,-3,0,3,6])
ax.set_ylim(0, 0.5)
ax.set_yticks(np.arange(1,6)*0.1)
ax.spines['left'].set_position(('data', 0))
ax.text(1.05, -0.03, r'$x$', ha='left',
        usetex=True, fontsize=25, color='k', 
        transform=ax.transAxes)
#%%
# create a figure updater
ud = UpdateFigure(ax, x, 'std', (1,2), 'g', (0.75,0.75))
anim = FuncAnimation(fig, ud, frames=121, blit=True)
anim.save(path/'evolve_moment1.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

ud = UpdateFigure(ax, x, 'std', (2,3), 'r', (0.75,0.62))
anim = FuncAnimation(fig, ud, frames=121, blit=True)
anim.save(path/'evolve_moment2.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

#! ============================================================
#! evolve mean
fig, ax = plt.subplots(1,1, figsize=(5,3.5), dpi=400)
x = np.linspace(-6, 6, 401)
y = norm.pdf(x, loc=0, scale=1)
ax.plot(x, y, color='b', zorder=1)
ax.text(0.75, 0.88, 
        r'$EX=$%2d'%(0.0), ha='left',usetex=True,
        fontsize=25, color='b', 
        transform=ax.transAxes)
ax.set_xlim(-6.0, 6.0)
ax.set_xticks([-6,-3,0,3,6])
ax.set_ylim(0, 0.5)
ax.set_yticks(np.arange(1,6)*0.1)
ax.spines['left'].set_position(('data', 0))
ax.text(1.05, -0.03, r'$x$', ha='left',
        usetex=True, fontsize=25, color='k', 
        transform=ax.transAxes)

# create a figure updater
ud = UpdateFigure(ax, x, 'mean', (0,2), 'g', (0.75,0.75))
anim = FuncAnimation(fig, ud, frames=121, blit=True)
anim.save(path/'evolve_moment3.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

ud = UpdateFigure(ax, x, 'mean', (0,-2), 'r', (0.75,0.62))
anim = FuncAnimation(fig, ud, frames=121, blit=True)
anim.save(path/'evolve_moment4.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
class UpdateFigure:
    def __init__(self, ax:plt.Axes, line, new_data:tuple, 
                 nframes:int, text:str, text_pos:tuple):
        self.ax = ax
        self.line = line
        self.new_x, self.new_y = new_data
        self.text_pos=text_pos
        self.text=text
        self.nframes = nframes
        # interpolation
        old_x = self.line.get_xdata()
        old_y = self.line.get_ydata()
        self.dx = (self.new_x - old_x) / self.nframes
        self.dy = (self.new_y - old_y) / self.nframes


    def update_gauss(self, y):
        self.line[0].set_data(self.x,y)

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i>0 and i < self.nframes+1:
            old_x = self.line.get_xdata()
            old_y = self.line.get_ydata()
            self.line.set_data(old_x + self.dx, old_y + self.dy)
        elif i == 101:
            self.ax.text(self.text_pos[0], self.text_pos[1], 
                         self.text, ha='left',usetex=True,
                         fontsize=25, color=self.line.get_color(), 
                         transform=self.ax.transAxes)
        else:
            pass
        return [self.line]
# %%
from scipy.stats import gamma
#! evolve mean
fig, ax = plt.subplots(1,1, figsize=(5,3.5), dpi=400, gridspec_kw={'left':0.05, 'right':0.91, 'top':0.95, 'bottom':0.15})
gamma_kws = {'loc':-np.sqrt(2), 'scale':1/np.sqrt(2), 'a':2}
x = np.linspace(-4, 4, 401)
y = gamma.pdf(x, **gamma_kws)
# mean, var, skew = gamma.stats(moments='mvs', **gamma_kws)
ax.plot(x, norm.pdf(x), color='b', zorder=1)[0].set_clip_on(False)
line, = ax.plot(x, norm.pdf(x), color='g', zorder=0)
line.set_clip_on(False)
text_kws = {'ha':'left', 'usetex':True, 'fontsize':25, 'transform':ax.transAxes}
ax.text(0.70, 0.88, r'$EX^3=0$', color='b', **text_kws)
# ax.plot(x, y, color='b', zorder=1)[0].set_clip_on(False)
# ax.plot(-x, y, color='r', zorder=1)[0].set_clip_on(False)
# ax.text(0.70, 0.75, r'$EX^3=1$', color='g', **text_kws)
# ax.text(0.70, 0.62, r'$EX^3=-1$', color='r', **text_kws)
ax.set_xlim(-4, 4)
ax.set_xticks([-4,-2,0,2,4])
ax.set_ylim(0, 0.6)
ax.set_yticks(np.arange(2,7,2)*0.1)
ax.spines['left'].set_position(('data', 0))
ax.text(1.05, -0.03, r'$x$', color='k', **text_kws)
plt.savefig(path/'test.pdf')
#%%
ud = UpdateFigure(ax, line, (x, y), 100, r'$EX^3=1$', (0.70,0.75))
anim = FuncAnimation(fig, ud, frames=121, blit=True)
anim.save(path/'evolve_moment5.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

line, = ax.plot(x, norm.pdf(x), color='r', zorder=0)
line.set_clip_on(False)
ud = UpdateFigure(ax, line, (x, np.flip(y)), 100, r'$EX^3=-1$', (0.70,0.62))
anim = FuncAnimation(fig, ud, frames=121, blit=True)
anim.save(path/'evolve_moment6.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
