#%%
from pathlib import Path
path = Path('./poisson/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
#%%
x = np.arange(0, 15)
y = poisson.pmf(x, 5)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x, y, fc='#6881B0', ec='k', width=1)
ax.set_xlabel('订单数量k', fontsize=25)
ax.set_ylabel('概率P(X=k)', fontsize=25)
ax.set_xlim(-0.5, 14.5)
fig.savefig(path/'poisson.pdf', dpi=300, bbox_inches='tight')
# %%
