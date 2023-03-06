#%%
from pathlib import Path
path = Path('./videos/poisson/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def _poisson(lam, x):
    s = np.zeros(x.shape)
    for i in range(len(x)):
        s[i] = np.power(lam,x[i])*np.exp(-lam)/np.math.factorial(x[i])
    return s

lam = 10
n = 100
Y = np.zeros([n])
X = np.arange(n)
poi = _poisson(lam, X)
kpk = np.arange(n) * poi
print(poi)
for z in range(n):
	Y[z] = 1100 * np.sum(kpk[:(z+1)]) - 1100 * z * np.sum(poi[:(z+1)]) + 1000 *z

print(Y)
print(np.argmax(Y))# 14
fig, ax = plt.subplots(1,1, figsize=(8,5),dpi=150)
ax.set_ylim(0,1e4)
ax.set_xlim(-1,100)
ax.set_xlabel('库存大小(z)', fontsize=30)
ax.set_ylabel('平均利润(Y)', fontsize=30)

ax.axvline(14, ymax=0.94, ls='--', color='#C00000')

ax.plot(X, Y)
ax.plot(14,Y[14],'o',color='r')
ax.text(50, 7400, r'平均销量 $\lambda$=10', fontsize=30)
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.annotate(r'$z^*$=14',xy=(14,Y[14]), 
             xytext=(16,Y[14]), ha='left', va='top',
             fontsize=30)
fig.savefig(path/'inventory_optim.pdf', bbox_inches='tight')

# %%
