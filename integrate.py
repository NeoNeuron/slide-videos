#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
path = Path('./lln/')
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

f = lambda x: np.exp(x**2)
fig, ax = plt.subplots(1,1, figsize=(8,6), )
x = np.linspace(0,1,100)
ax.plot(x, f(x))
ax.set_xlabel(r'$x$', fontsize=26, usetex=True)
ax.set_ylabel(r'$f(x)$', fontsize=26, usetex=True)
ax.text(0.45, 0.5, r'$f(x)=e^{x^2}$',
        fontsize=30, usetex=True, transform=ax.transAxes)
ylim = ax.get_ylim()
ax.plot([0,0], [1,0.5],ls='--', color='k')
ax.plot([1,1], [f(1),0.5],ls='--', color='k')
ax.set_ylim(ylim)
plt.tight_layout()
fig.savefig(path/'exp.png', dpi=300)