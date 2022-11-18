#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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
plt.savefig('./function_of_random_variables/incubation.pdf')
#%%