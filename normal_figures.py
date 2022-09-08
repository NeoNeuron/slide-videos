# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams['font.size']=20
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
GREEN = '#00B050'
RED   = '#C00000'
# %%
x = np.linspace(0, 6000, 6000)
rv_y_ok = norm.rvs(loc=3785, scale=541, size=1000, random_state=0)
rv_y_nok = norm.rvs(loc=1860, scale=240, size=1000, random_state=11)
y_ok = norm.pdf(x, loc=3785, scale=541)
y_nok = norm.pdf(x, loc=1860, scale=240)

# draw figures
fig, ax = plt.subplots(1,1,figsize=(10,3), 
                       gridspec_kw={'top':0.95,'bottom':0.22,'left':0.12,'right':0.99,})
ax.hist(rv_y_ok,  range=(0,6000), bins=100, density=True, alpha=0.75, color=GREEN)
ax.hist(rv_y_nok, range=(0,6000), bins=100, density=True, alpha=0.75, color=RED)
ax.set_ylim(0)
ax.set_xlim(0,6000)
ax.set_xticks(np.linspace(1860*2-3785, 2*3785-1860, 10))
ax.set_xticklabels(['', '', '', r'$\mu_{NOK}$', '', '', r'$\mu_{OK}$', '', '', ''], color='w')
ax.set_yticks([0,0.001,0.002])
ax.set_ylabel('概率')
ax.set_xlabel('白色像素个数')
plt.savefig('normal_fig_without_pdf.png', dpi=300)

ax.plot(x, y_ok,  c=GREEN, zorder=10, lw=4)
ax.plot(x, y_nok, c=RED  , zorder=10, lw=4)
ax.set_xticklabels(['', '', '', r'$\mu_{NOK}$', '', '', r'$\mu_{OK}$', '', '', ''], color='k')
plt.savefig('normal_fig_with_pdf.png', dpi=300)

# %%
fig, ax = plt.subplots(1,1,figsize=(10,3), 
                       gridspec_kw={'top':0.95,'bottom':0.22,'left':0.12,'right':0.99,})
ax.plot(x, y_ok,  c=GREEN, zorder=10, lw=2)
ax.plot(x, y_nok, c=RED  , zorder=10, lw=2)
ax.set_ylim(0)
ax.set_xlim(0,6000)
ax.set_xticks(np.linspace(1860*2-3785, 2*3785-1860, 10))
ax.set_xticklabels(['', '', '', r'$\mu_{NOK}$', '', '', r'$\mu_{OK}$', '', '', ''], color='k')
ax.set_yticks([0,0.001,0.002])
ax.set_ylabel('概率')
ax.set_xlabel('白色像素个数')
plt.savefig('normal_fig_pdf0.png', dpi=300)
ax.axvline(2120, zorder=10, lw=3, color='navy')
ax.fill_between(x[x>2120], 0, y_nok[x>2120], color='navy', alpha=0.7)
plt.savefig('normal_fig_pdf1.png', dpi=300)
ax.axvline(2900, zorder=10, lw=3, color='orange')
ax.fill_between(x[x<2900], 0, y_ok[x<2900], color='orange', alpha=0.7)
plt.savefig('normal_fig_pdf2.png', dpi=300)
#%%
fig, ax = plt.subplots(1,1,figsize=(10,3), 
                       gridspec_kw={'top':0.95,'bottom':0.22,'left':0.12,'right':0.99,})
ax.plot(x, y_ok,  c=GREEN, zorder=10, lw=2)
ax.plot(x, y_nok, c=RED  , zorder=10, lw=2)
ax.set_ylim(0)
ax.set_xlim(0,6000)
ax.set_xticks(np.linspace(1860*2-3785, 2*3785-1860, 10))
ax.set_xticklabels(['', '', '', r'$\mu_{NOK}$', '', '', r'$\mu_{OK}$', '', '', ''], color='k')
ax.set_yticks([0,0.001,0.002])
ax.set_ylabel('概率')
ax.set_xlabel('白色像素个数')
ax.axvline(3260, zorder=10, lw=3, color='orange')
ax.fill_between(x[x<3260], 0, y_ok[x<3260], color='orange', alpha=0.7)
plt.savefig('normal_fig_pdf_cdf.png', dpi=300)

# %%
x = np.linspace(0, 6000, 6000)
y_ok = norm.pdf(x, loc=4585, scale=441)
y_nok = norm.pdf(x, loc=1260, scale=200)

fig, ax = plt.subplots(1,1,figsize=(10,3), 
                       gridspec_kw={'top':0.95,'bottom':0.22,'left':0.12,'right':0.99,})
ax.plot(x, y_ok,  c=GREEN, zorder=10, lw=2)
ax.plot(x, y_nok, c=RED  , zorder=10, lw=2)
ax.set_ylim(0)
ax.set_xlim(0,6000)
xticks = np.linspace(1260, 4585, 6)
for i in range(2):
    xticks = np.insert(xticks, 0, xticks[0]*2-xticks[1])
    xticks = np.append(xticks, [xticks[-1]*2-xticks[-2]])
ax.set_xticks(xticks)
ax.set_xticklabels(['', '', r'$\mu_{NOK}$', '', '', '', '', r'$\mu_{OK}$', '', ''], color='k')
ax.set_yticks([0,0.001,0.002])
ax.set_ylabel('概率')
ax.set_xlabel('白色像素个数')
plt.savefig('normal_fig_pdf3.png', dpi=300)
ax.axvline(2620, zorder=10, lw=3, color='navy')
plt.savefig('normal_fig_pdf4.png', dpi=300)
# %%
