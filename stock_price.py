#%%
from pathlib import Path
path = Path('./covariance/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["axes.labelsize"] = 28
# plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
# %%

stock_x = np.array([81.86, 93.38, 79.13, 90.15, 79.61, 76.86, 95.30, 74.91, 63.31, 81.44,])
stock_y = np.array([30.47, 33.3, 32.41, 39.39, 30.4, 35.25, 36.65, 27.53, 19.62, 30.73])
stock_z = np.array([83.18 ,85.14 ,78.51 ,71.13 ,75.04 ,78.92 ,83.08 ,83.46 ,91.85 ,79.77])
stock_w = np.array([28.17, 24.24, 32.45, 31.08, 26.42, 30.28, 28.74, 25.7, 25, 30.99])
#%%
fig, ax = plt.subplots(2,1, figsize=(6.5,6),
                       gridspec_kw=dict(left=0.2, right=0.95, bottom=0.15, top=0.95, hspace=0.4))

ax[0].plot(np.arange(1,11), stock_x, '-o',ms=15, markeredgewidth=2.5, markerfacecolor='#353A71', markeredgecolor='#A7BED2')
ax[0].set_xlabel('时间')
ax[0].set_ylabel('$X$股价')
ax[0].set_ylim(60,100)
ax[1].plot(np.arange(1,11), stock_y, '-o',ms=15, markeredgewidth=2.5, markerfacecolor='#353A71', markeredgecolor='#A7BED2')
ax[1].set_xlabel('时间')
ax[1].set_ylabel('$Y$股价')
ax[1].set_ylim(10,50)
plt.savefig(path/'stock_xy_sep.pdf')
fig, ax = plt.subplots(1,1, figsize=(6.5,6),
                       gridspec_kw=dict(left=0.2, right=0.95, bottom=0.15, top=0.95))
ax.plot(stock_x, stock_y, 'o',ms=15, markeredgewidth=2.5, markerfacecolor='#C00000', markeredgecolor='#D77F66')
ax.set_xlabel('$X$股价')
ax.set_xlim(60,100)
ax.set_ylabel('$Y$股价')
ax.set_ylim(10,50)
ax.set_yticks(np.linspace(10,50,5))
pval = np.polyfit(stock_x, stock_y, deg=1)
ax.plot(np.sort(stock_x), np.polyval(pval, np.sort(stock_x)), '--', c='#C00000')
plt.savefig(path/'stock_xy_cov.pdf')
#%%
fig, ax = plt.subplots(2,1, figsize=(6.5,6),
                       gridspec_kw=dict(left=0.2, right=0.95, bottom=0.15, top=0.95, hspace=0.4))

ax[0].plot(np.arange(1,11), stock_z, '-o',ms=15, markeredgewidth=2.5, markerfacecolor='#353A71', markeredgecolor='#A7BED2')
ax[0].set_xlabel('时间')
ax[0].set_ylabel('$Z$股价')
ax[0].set_ylim(60,100)
ax[1].plot(np.arange(1,11), stock_w, '-o',ms=15, markeredgewidth=2.5, markerfacecolor='#353A71', markeredgecolor='#A7BED2')
ax[1].set_xlabel('时间')
ax[1].set_ylabel('$W$股价')
ax[1].set_ylim(10,50)
plt.savefig(path/'stock_zw_sep.pdf')
fig, ax = plt.subplots(1,1, figsize=(6.5,6),
                       gridspec_kw=dict(left=0.2, right=0.95, bottom=0.15, top=0.95))
ax.plot(stock_z, stock_w, 'o',ms=15, markeredgewidth=2.5, markerfacecolor='#C00000', markeredgecolor='#D77F66')
ax.set_xlabel('$Z$股价')
ax.set_xlim(70,94)
ax.set_ylabel('$W$股价')
ax.set_ylim(22,34)
# ax.set_yticks(np.linspace(10,50,5))
pval = np.polyfit(stock_z, stock_w, deg=1)
ax.plot(np.sort(stock_z), np.polyval(pval, np.sort(stock_z)), '--', c='#C00000')
plt.savefig(path/'stock_zw_cov.pdf')