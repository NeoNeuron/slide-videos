# %%
PATH='point_estimation/'
import os
os.makedirs(PATH, exist_ok=True)
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import expon
plt.rcParams['font.size']=30
# %%
fig, ax = plt.subplots(1,1, figsize=(8,5), dpi=200,
                       gridspec_kw={'left':0.15, 'right':0.95, 'top':0.95, 'bottom':0.2})
x = np.linspace(0, 10, 101)
ax.plot(x, expon.pdf(x, scale=0.5), lw=3, label=r'$\theta=0.5$')
ax.plot(x, expon.pdf(x, scale=1.0), lw=3, label=r'$\theta=1.0$')
ax.axvline(expon.stats(scale=0.5, moments='m'), ls='--')
ax.axvline(expon.stats(scale=1.0, moments='m'), ls='--', color='tab:orange')
ax.set_xlim(0, 6)
ax.set_ylim(0)
ax.legend()
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
ax.set_xlabel('使用寿命(万小时)')
ax.set_ylabel('概率密度')
fig.savefig(PATH+'bulb_lifetime.png', dpi=300)
#%%