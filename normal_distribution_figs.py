#%%
from pathlib import Path
path = Path('./normal_distribution/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
#%%
fig, ax = plt.subplots(
    1,1, figsize=(8,5),
    gridspec_kw=dict(left=0.2, right=0.90, bottom=0.1, top=0.90))

x = np.linspace(-4,4,101)
ax.set_ylim(0,0.5)
ax.set_xlim(-4,4)
ax.set_xticks(np.linspace(-4,4,5))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.text(-0.05,1, '$f(x)$', 
        fontsize=30,
        va='bottom', ha='right', transform=ax.transAxes)
ax.text(1.08,-0.05, '$x$', 
        fontsize=30,
        va='bottom', ha='right', transform=ax.transAxes)
fig.savefig(path/'frame1.pdf')
ax.plot(x, norm.pdf(x), c='r', lw=2)[0].set_clip_on(False)
fig.savefig(path/'frame2.pdf')
#%%
fig, ax = plt.subplots(
    1,1, figsize=(8,4.5),
    gridspec_kw=dict(left=0.2, right=0.90, bottom=0.1, top=0.90))

x = np.linspace(-4,4,401)
ax.set_ylim(0,0.4)
ax.set_xlim(-4,4)
ax.set_xticks(np.linspace(-4,4,5))
ax.set_yticks(np.linspace(0,0.4,6))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.text(-0.05,1, '$f(x)$', 
        fontsize=30,
        va='bottom', ha='right', transform=ax.transAxes)
ax.text(1.08,-0.05, '$x$', 
        fontsize=30,
        va='bottom', ha='right', transform=ax.transAxes)
ax.plot(x, norm.pdf(x), lw=2, c='navy')[0].set_clip_on(False)
fig.savefig(path/'3sigma_frame1.png', dpi=300)
x1_1 = np.linspace(-4,1,301)
shade1 = ax.fill_between(x1_1, 0, norm.pdf(x1_1), color='r', alpha=0.25, edgecolor='none')
fig.savefig(path/'3sigma_frame1_1.png', dpi=300)
x1_2 = np.linspace(-4,-1,201)
shade2 = ax.fill_between(x1_2, 0, norm.pdf(x1_2), color='g', alpha=0.25, edgecolor='none')
fig.savefig(path/'3sigma_frame1_2.png', dpi=300)
shade1.remove()
shade2.remove()
x1 = np.linspace(-1,1,101)
ax.fill_between(x1, 0, norm.pdf(x1), color='r', alpha=0.25, edgecolor='none')
fig.savefig(path/'3sigma_frame2.png', dpi=300)
x2 = np.linspace(-2,2,201)
ax.fill_between(x2, 0, norm.pdf(x2), color='r', alpha=0.25, edgecolor='none')
fig.savefig(path/'3sigma_frame3.png', dpi=300)
x3 = np.linspace(-3,3,301)
ax.fill_between(x3, 0, norm.pdf(x3), color='r', alpha=0.25, edgecolor='none')
fig.savefig(path/'3sigma_frame4.png', dpi=300)
# %%
