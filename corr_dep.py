# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"] = False
from pathlib import Path
path = Path('./covariance/')
path.mkdir(exist_ok=True)
#%%
BLUE = '#000089'
N = 1000
theta = np.random.rand(N)*2*np.pi
rho = np.random.randn(N)+10

x = rho*np.cos(theta)
y = rho*np.sin(theta)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(x, y, 'o', color=BLUE, alpha=1.0)
ax.axis('scaled')
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(path/'corr_dep.png', dpi=300, bbox_inches='tight', transparent=True)
# %%
