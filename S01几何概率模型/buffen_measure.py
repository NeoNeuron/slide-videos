# %%
from init import *
from mpl_toolkits.axisartist.axislines import AxesZero
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
plt.rcParams['font.size']=30
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
plt.rcParams['text.usetex']=True
# %%
prefix = 'buffen_int'
fig = plt.figure()
gs = plt.GridSpec(1, 1, top=0.9, bottom=0.1, left=0.1, right=0.9)
ax = fig.add_subplot(gs[0], axes_class=AxesZero)
ax.set_xlim(0, 1.2*np.pi)
ax.set_ylim(0, 1.2)
ax.set_xticks([])
ax.set_yticks([])
ax.text(-0.17,-0.14,r'$O$', )
ax.text(0.1,1.25,r'$x$', )
ax.text(1.22*np.pi,-0.15,r'$\theta$', )
for direction in ["xzero", "yzero"]:
    ax.axis[direction].set_axisline_style("-|>", )
    ax.axis[direction].set_visible(True)
for direction in ['right', 'top', 'bottom', 'left']:
    ax.axis[direction].set_visible(False)
plt.savefig(path/prefix+'_step1.pdf')

ax.set_xticks([0, np.pi])
ax.set_xticklabels(['', r'$\pi$'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['', r'$\frac{a}{2}$'])
ax.plot([0,np.pi], [1,1], c='k')
ax.plot([np.pi,np.pi], [1,0], c='k')
ax.text(0.28,0.54,r'$\Omega$', color='#93391E')

plt.savefig(path/prefix+'_step2.pdf')

ax.text(0.30*np.pi,0.8,r'$x=\frac{l}{2}\sin\theta$', )
x = np.linspace(0,np.pi,100)
y = 0.7*np.sin(x)
ax.plot(x,y, color='navy')
plt.savefig(path/prefix+'_step3.pdf')

ax.fill_between(x, 0, y, alpha=0.5)
ax.text(1.4,0.34,r'$A$', color='#93391E')
plt.savefig(path/prefix+'_step4.pdf')
# %%
