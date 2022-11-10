#%%
from pathlib import Path
PATH = Path('LinearRegression/')
PATH.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
def create_canvas():
    fig, ax = plt.subplots(1,1, figsize=(6,5), 
                           gridspec_kw={'top':0.95, 'bottom':0.22, 'left':0.22, 'right':0.98})
    # Move left and bottom spines outward by 10 points
    ax.spines.left.set_position(('outward', 15))
    ax.spines.bottom.set_position(('outward', 15))
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return fig, ax
#%%
np.random.seed(2022)
x = np.array([157, 158, 163, 165,167,167,168,169, 170, 172])
y=1.1*x-20.5 + np.random.randn(x.shape[0])*2
# %%
fig, ax = create_canvas()
ax.plot(x, y, 'o', ms=15, markerfacecolor='royalblue', markeredgecolor='orange')
ax.set_xlabel('父母平均身高(cm)', fontsize=30)
ax.set_ylabel('女孩成年后身高(cm)', fontsize=30)
ax.set_ylim(151, 171)
fig.savefig(PATH/'parents_offspring_height1.png', dpi=200)
p = np.polyfit(x, y, 1)
line, = ax.plot([x.min(),x.max()], np.polyval(p, [x.min(), x.max()]), 'r')
fig.savefig(PATH/'parents_offspring_height2.png', dpi=200)
x_range = np.array([156, 210])
line.set_data(x_range, 5.01+0.95*x_range)
line.set_color('royalblue')
ax.set_xlim(155, 210)
ax.set_ylim(151, 205)
ax.scatter([208],[203],marker='^',s=140,c='r', zorder=10)
ax.plot([208, 208],[148, 203],'--',c='r', zorder=10)[0].set_clip_on(False)
ax.plot([153, 208],[203, 203],'--',c='r', zorder=10)[0].set_clip_on(False)
fig.savefig(PATH/'parents_offspring_height3.png', dpi=200)

# %%
np.random.seed(509)
y=1*x-100 + np.random.randn(x.shape[0])*2
fig, ax = create_canvas()
ax.plot(x, y, 'o', ms=15, markerfacecolor='royalblue', markeredgecolor='orange')
ax.set_xlabel('身高(cm)', fontsize=30)
ax.set_ylabel('体重(kg)', fontsize=30)
ax.set_ylim(54, 74)
fig.savefig(PATH/'height_weight1.png', dpi=200)
p = np.polyfit(x, y, 1)
ax.plot([x.min(),x.max()], np.polyval(p, [x.min(), x.max()]), 'r')
fig.savefig(PATH/'height_weight2.png', dpi=200)

# %%

np.random.seed(11)
x = np.random.randn(10)*20+100
y=5*(x + np.random.randn(x.shape[0])*6)
fig, ax = create_canvas()
ax.plot(x, y, 'o', ms=15, markerfacecolor='royalblue', markeredgecolor='orange')
ax.set_xlabel('住房面积($m^2$)', fontsize=30)
ax.set_ylabel('房价(万)', fontsize=30)
ax.set_ylim(230, 680)
fig.savefig(PATH/'house_price1.png', dpi=200)
p = np.polyfit(x, y, 1)
ax.plot([x.min(),x.max()], np.polyval(p, [x.min(), x.max()]), 'r')
fig.savefig(PATH/'house_price2.png', dpi=200)
#%%