# %%
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats import multivariate_normal
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 16
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
###patch end###

#%%
mean = np.array([81.5, 31.5, 81, 28.3])
cov = np.diag([9.50, 5.45, 5.72, 2.87])**2
cov[0,1]=cov[1,0]=0.8*9.50*5.45
cov[2,3]=cov[3,2]=-0.8*5.72*2.87
data = multivariate_normal.rvs(mean, cov, size=10000, random_state=2022)
H, _x, _y = np.histogram2d(data[:,0], data[:,1], bins=20)
H /=10000
dx = _x[1]-_x[0]
dy = _y[1]-_y[0]
_xx, _yy = np.meshgrid(_x[:-1], _y[:-1])
N = _xx.flatten().shape[0]
dx = np.tile(np.ones(_x.shape[0]-1)*dx, (_y.shape[0]-1,1))
dy = np.tile(np.ones(_y.shape[0]-1)*dy, (_x.shape[0]-1,1)).T
#%%
fig = plt.figure(figsize=(8,6),dpi=200)
ax = fig.add_subplot(projection='3d')
ax.invert_yaxis()
bars = ax.bar3d(
    _xx.flatten(), _yy.flatten(), np.zeros(N), 
    dx.flatten(), dy.flatten(),H.flatten(),
    color=plt.cm.jet(H.flatten()/H.max()),
    shade=False, zsort='max',
    )
ax.set_xlabel('X股票价格')
ax.set_ylabel('Y股票价格')
ax.set_xlim(_x[0], _x[-1])
ax.set_ylim(_y[0], _y[-1])
ax.set_zticks(np.arange(5)*0.01)
ax.set_zlabel('概率', rotation=270)
ax.view_init(15, 180-55)
plt.savefig('3d_bar.pdf', transparent=True)
# %%
print(data.std(0))
# %%
print(np.mean(data[:,2]*data[:,3]))
# %%
