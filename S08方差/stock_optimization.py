#%%
from init import *
x1 = np.arange(1800)
x2 = np.arange(2400)
x1, x2 = np.meshgrid(x1, x2)
mask = 100000-60*x1-48*x2<0
reward = 6*x1+4*x2+0.036*(100000-60*x1-48*x2)
reward[mask]=0
plt.pcolormesh(x1, x2, reward)
plt.xlabel('stock 1')
plt.ylabel('stock 2')
plt.colorbar()
#%%
DR = 55*x1**2+28*x2**2
plt.plot(DR[~mask], reward[~mask], '.', )
# %%
