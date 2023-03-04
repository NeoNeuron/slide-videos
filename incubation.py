#%%
from pathlib import Path
path = Path('./function_of_random_variables/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
#%%
x = np.linspace(-3,3,1001)
y = norm.pdf(x, loc=1.51, scale=0.4)
x_linear = np.exp(x)
x_linear = np.floor(x_linear)
y_discrete = np.zeros(15)
for i in range(15):
    y_discrete[i] = y[x_linear==i].mean()
y_discrete /= y_discrete.sum() 
plt.figure(figsize=(6,3.5))
plt.bar(np.arange(15), y_discrete, align='center')
plt.ylabel('频率', fontsize=20)
plt.xlabel('天数', fontsize=20)
plt.xlim(-0.5, 14.5)
plt.ylim(0,0.23)
plt.tight_layout()
plt.savefig(path/'incubation.pdf')
#%%
x = np.linspace(0,14,1001)
y = norm.pdf(x, loc=1.51, scale=0.4)
x_linear = np.exp(x)
plt.figure(figsize=(6,3))
plt.bar(np.arange(15), y_discrete, align='center', width=0.8)
plt.plot(x_linear-0.5, y/5, c='orange', lw=3)
plt.ylabel('频率', fontsize=20)
plt.xlabel('天数', fontsize=20)
plt.xlim(-0.5, 14.5)
plt.ylim(0,0.23)
plt.xticks(np.arange(0,15,2), labels=['0','','4','','8','','12',''])
plt.tight_layout()
plt.savefig(path/'incubation_fit.pdf')
#%%
counter = 1

def savefig(fig):
    global counter
    fig.savefig(path/f'virus_{counter}.pdf')
    counter += 1
fig, ax = plt.subplots(figsize=(6,3.5), gridspec_kw={'top':0.85})
xlim = [0,5]
ylim = [0,1]
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xticks([])
ax.set_yticks([])
ax.scatter(xlim[-1], ylim[0], s=180, color='k', marker='>', ).set_clip_on(False)
ax.scatter(xlim[0], ylim[-1], s=180, color='k', marker='^', ).set_clip_on(False)
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(3)
ax.text(0, 1.1, '病毒数量', color='k', ha='center', va='center', fontsize=20, transform=ax.transAxes, clip_on=False)
ax.text(1., -0.1, '时间', color='k', ha='center', va='center', fontsize=20, transform=ax.transAxes, clip_on=False)
savefig(fig)
ax.text(-0.25, 0.3, r'$Z$', color='k', ha='center', va='center', fontsize=20, clip_on=False)
ax.plot(0, 0.3, 'o', ms=10, mec='#C00000', mfc='#E9AA94', clip_on=False, zorder=10)
savefig(fig)
x = np.linspace(0,5,101)
ax.plot(x, 0.3*np.exp(x/4), lw=3, c='#C00000')
savefig(fig)
ax.text(-0.25, 0.8, r'$\overline{M}$', color='k', ha='center', va='center', fontsize=20, clip_on=False)
ax.plot(0, 0.8, 'o', ms=10, mec='#2E5597', mfc='#B4C7E7', clip_on=False, zorder=10)
savefig(fig)
intersect = 4*np.log(0.8/0.3)
ax.plot([0, intersect], [0.8, 0.8], '--', lw=3, c='#2E5597')
savefig(fig)
ax.plot([intersect, intersect], [0.8, 0], '--', lw=3, c='#2E5597')
savefig(fig)
ax.text(intersect, -0.1, r'$X$', color='k', ha='center', va='center', fontsize=20, clip_on=False)
savefig(fig)
# %%
