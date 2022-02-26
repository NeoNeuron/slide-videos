#%%
import numpy as np
import matplotlib.pyplot as plt
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
#%%

np.random.seed(20220225)
X = np.random.multivariate_normal(np.zeros(2), np.array([[1,0.6],[0.6,1]]), size=2000)
X *= 15
X += 50
X = np.floor(X)


fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.plot(X[:,0], X[:,1], '.', ms=5, alpha=0.5)
ax.axis('scaled')
ax.set_xlim(0,100)
ax.set_ylim(0,100)
plt.savefig('cov_fig.pdf', transparent=True)
# %%
