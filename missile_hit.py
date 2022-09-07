#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

# %%
class UpdateFigure:
    def __init__(self, ax:np.ndarray, data:np.ndarray):
        """Plot the first frame for the animation.

        Args:
            ax (np.ndarray): array of axes
            data (np.ndarray): 1-D array of number of passagers for each days
        """

        self.ax= ax
        self.data = data
        p=0.01
        self.x, self.y = np.where(data<p)
        self.x += 1
        trials = np.sum(data<p, axis=1, dtype=bool)
        self.hit_number = np.cumsum(trials.astype(float))
        self.x_grid = np.arange(data.shape[0])+1
        self.hit_rate = self.hit_number/self.x_grid

        hit_theory = 1-(1-p)**data.shape[1]

        self.dots, = self.ax[0].plot([],[],'o',color='r', markeredgecolor='k', ms=8, zorder=100)

        self.rects = self.ax[1].barh([1, 0], [0, 0], color=['r','grey'])
        self.texts = [self.ax[1].text(x-0.2, y, '', color='w', fontweight='bold', fontsize=20, ha='right', va='center') for y, x in zip([1,0], [0,0])]
        self.ax[1].spines['top'].set_visible(False)
        self.ax[1].spines['bottom'].set_visible(False)
        self.ax[1].spines['right'].set_visible(False)
        self.ax[1].set_yticks([0,1])
        self.ax[1].set_yticklabels(['总次数','命中次数'], fontsize=15)

        self.ax[2].axhline(hit_theory, c='m')
        self.line, = self.ax[2].plot([], [], '-o', c='navy', markerfacecolor='orange', )
        self.line.set_clip_on(False)


    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0 and i<=self.data.shape[0]:
            # update lines
            mask = self.x<=i
            self.dots.set_data(self.x[mask], self.y[mask])
            self.ax[0].axvline(i,color='grey', alpha=0.4)
            self.line.set_data(self.x_grid[:i], self.hit_rate[:i])
            
            # update the height of bars for histogram
            for rect, h in zip(self.rects, [self.hit_number[i-1], self.x_grid[i-1]]):
                rect.set_width(h)
            for text, x in zip(self.texts, [self.hit_number[i-1], self.x_grid[i-1]]):
                if x > 1:
                    text.set_x(x-0.2)
                    text.set_text(f'{x:.0f}')
        return self.rects

# setup contents
fig, ax = plt.subplots(3,1,figsize=(10,10), 
                       gridspec_kw={'height_ratios':[4,1,4], 'top':0.95,
                                    'bottom':0.10, 'left':0.1, 'right':0.95,
                                    'hspace':0.15})
ax[0].set_ylabel('发射的子弹数', fontsize=20)
ax[0].set_xticklabels([])
ax[2].set_ylabel('理论命中率和试验射命中率', fontsize=20)
ax[2].set_xlabel('试验次数', fontsize=20)
[axi.set_xlim(0,50) for axi in ax]
ax[0].set_ylim(0,100)
ax[2].set_ylim(0,1)

np.random.seed(202)
data = np.random.rand(50, 100)

# create a figure updater
ud = UpdateFigure(ax, data)
plt.savefig('test.pdf')
#%%
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=52, blit=True)
# save animation as *.mp4
anim.save('missile_hit.mp4', fps=12, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
# plot figure: number of missile v.s. shot precision
fig, ax = plt.subplots(1,1, figsize=(8,6))
p=0.01
x = np.arange(801)
y = 1-(1-p)**x

ax.plot(x,y,lw=3,c='navy')[0].set_clip_on(False)
ax.set_xlabel('发射子弹数')
ax.set_ylabel('命中率')
ax.set_xlim(0,800)
ax.set_ylim(0,1)

for x_, color_ in zip([100,459], ['m', 'r']):
    y_ = 1-(1-p)**x_
    ax.plot([x_,], [y_,], 'o', color=color_, ms=10, zorder=10, markeredgecolor='navy')[0].set_clip_on(False)
    ax.plot([x_,x_], [0, y_], '--', color=color_)
    ax.plot([0,x_], [y_, y_], '--', color=color_)
    ax.text(x_+5, 0, f'{x_:.0f}', color=color_, ha='left', va='bottom')
    ax.text(0, y_+0.01, f'{y_:6.3f}', color=color_, ha='left', va='bottom')

plt.savefig('missile_hit.pdf')