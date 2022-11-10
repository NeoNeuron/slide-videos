# %%
from pathlib import Path
PATH = Path('LinearRegression/')
PATH.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
from sklearn.linear_model import LinearRegression
#%%
#300 data
np.random.seed(2022)
f = np.array([157, 158, 163, 165,167,167,168,169,170, 172])
n = f.shape[0]
data1=22/20*(f-155) + np.random.normal(0,2,n) + 150

model = LinearRegression()
model.fit(f.reshape(-1,1), data1)
b1 = model.coef_[0]
b0 = model.intercept_

fig, ax = plt.subplots(1,1,figsize=(8,6),
                       gridspec_kw = dict(left=0.15, right=0.88, top=0.9, bottom=0.15)
                       )
ax.set_xlabel('父母平均身高(cm)', fontsize=30)
ax.set_ylabel('女孩成年后身高(cm)', fontsize=30)
ax.set_ylim([150,220])
ax.set_xlim([155,210]) 
ax.set_xticks([155+i*10 for i in range(6)])
ax.set_yticks([150+i*10 for i in range(8)])

ax.plot(f, data1, 'o', ms=8, c='mediumblue')

sigma2 = np.sum((data1-b0-b1*f)**2)/(n-2)*2.5
def get_width(x0):
	t = 1.8595
	return t*np.sqrt(1+1/n + (x0-np.mean(f))**2/np.sum((f-np.mean(f))**2)*sigma2)
X = np.linspace(156,208,100)
Y = np.zeros([100])
for i in range(100):
	Y[i] = get_width(X[i])
fig.savefig(PATH/'lr_height_1.png', dpi=300)
b0 = 5.01
b1 = 0.95
ax.plot([156,208], b1*np.array([156,208])+b0, c='mediumblue')
fig.savefig(PATH/'lr_height_2.png', dpi=300)

def CI(x, b0, b1):
	width = get_width(x)
	l1 = ax.axhline(b1*x+b0+width, xmax=(x-155)/(210-155), ls='--', color='r')
	l2 = ax.axhline(b1*x+b0-width, xmax=(x-155)/(210-155), ls='--', color='r')
	lines=ax.errorbar([x],[b1*x+b0], yerr=width, capsize=8, marker='^',ms=10,c='r', alpha=0.8, zorder=10)
	ax.text(x+1,b1*x+b0, f'({x:.0f}, {b1*x+b0:.0f})', fontsize=15, ha='left', va='top')
	text1 = ax.text(156,b1*x+b0+width+0.3, f'{np.ceil(b1*x+b0+width):.0f}', fontsize=12, ha='left', va='bottom', color='r', alpha=0.8)
	text2 = ax.text(156,b1*x+b0-width-0.4, f'{np.ceil(b1*x+b0-width):.0f}', fontsize=12, ha='left', va='top', color='r', alpha=0.8)
	return [l1, l2, lines.lines[1][0], lines.lines[1][1], lines.lines[2][0], text1, text2]

lines =CI(208, b0, b1)
fig.savefig(PATH/'lr_height_3.png', dpi=300)
[line.set_color('w') for line in lines]
lines =CI(165, b0, b1)
fig.savefig(PATH/'lr_height_4.png', dpi=300)
[line.set_color('w') for line in lines]
ax.plot(X, b1*X+b0+Y, ls='--',color = '#1f77b4')
ax.plot(X, b1*X+b0-Y, ls='--',color = '#1f77b4')
fig.savefig(PATH/'lr_height_5.png', dpi=300)
# %%
