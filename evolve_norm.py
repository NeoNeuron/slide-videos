# %%
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
                text = r'$\mu$ = %2d'%self.var_range[1]
            elif self.var_name == 'std':
                text = r'$\sigma$ = %2d'%self.var_range[1]
            self.ax.text(self.text_pos[0], self.text_pos[1], 
                         text, ha='left',
                         fontsize=30, color=self.c, 
                         transform=self.ax.transAxes)
        else:
            pass
        return self.line

#! ============================================================
#! evolve std
fig, ax = plt.subplots(1,1, dpi=400)
x = np.linspace(-6, 6, 401)
y = norm.pdf(x, loc=0, scale=1)
ax.plot(x, y, color='b', zorder=1)
ax.text(0.78, 0.88, 
        r'$\sigma$ = %2d'%(1.0), ha='left',
        fontsize=30, color='b', 
        transform=ax.transAxes)
ax.set_xlim(-6.0, 6.0)
ax.set_xticks([-6,-3,0,3,6])
ax.set_ylim(0, 0.5)
ax.spines['left'].set_position(('data', 0))

# create a figure updater
ud = UpdateFigure(ax, x, 'std', (1,2), 'g', (0.78,0.75))
anim = FuncAnimation(fig, ud, frames=121, blit=True)
anim.save('evolve_norm1.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

ud = UpdateFigure(ax, x, 'std', (2,3), 'r', (0.78,0.62))
anim = FuncAnimation(fig, ud, frames=121, blit=True)
anim.save('evolve_norm2.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

#! ============================================================
#! evolve mean
fig, ax = plt.subplots(1,1, dpi=400)
x = np.linspace(-6, 6, 401)
y = norm.pdf(x, loc=0, scale=1)
ax.plot(x, y, color='b', zorder=1)
ax.text(0.78, 0.88, 
        r'$\mu$ = %2d'%(0.0), ha='left',
        fontsize=30, color='b', 
        transform=ax.transAxes)
ax.set_xlim(-6.0, 6.0)
ax.set_xticks([-6,-3,0,3,6])
ax.set_ylim(0, 0.5)
ax.spines['left'].set_position(('data', 0))

# create a figure updater
ud = UpdateFigure(ax, x, 'mean', (0,2), 'g', (0.78,0.75))
anim = FuncAnimation(fig, ud, frames=121, blit=True)
anim.save('evolve_norm3.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

ud = UpdateFigure(ax, x, 'mean', (0,-2), 'r', (0.78,0.62))
anim = FuncAnimation(fig, ud, frames=121, blit=True)
anim.save('evolve_norm4.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%