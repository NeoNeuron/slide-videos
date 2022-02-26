# %%
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
data = np.loadtxt('data.csv', delimiter='\t')
_x = np.arange(5)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
#%%
fig = plt.figure(figsize=(8,6),dpi=200,)
ax = fig.add_subplot(projection='3d')
ax.bar3d(_xx.flatten(), _yy.flatten(), np.zeros(25), np.ones(25),np.ones(25),data.flatten(),color=plt.cm.rainbow(data.flatten()/data.max()))
ax.set_ylabel('高数成绩X')
ax.set_xlabel('线性代数成绩Y')
ax.set_zlabel('相关系数', rotation=270)
plt.savefig('3d_bar.pdf', transparent=True)
# %%
