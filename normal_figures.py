# %%
from pathlib import Path
path = Path('./normal_distribution/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams['font.size']=20
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
GREEN = '#00B050'
RED   = '#C00000'

def create_fig():
    fig, ax = plt.subplots(1,1,figsize=(10,3), 
                        gridspec_kw={'top':0.91,'bottom':0.20,'left':0.06,'right':0.99,})
    ax.set_xticks(np.arange(5)*1000)
    ax.set_yticks(np.arange(4)*1e-3)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
    ax.set_ylabel('概率')
    ax.set_xlabel('白色像素个数')
    ax.set_xlim(0,4500)
    ax.set_ylim(0, 3e-3)
    return ax
# %%
# Generate random numbers
x = np.linspace(0, 4500, 4501)
mu = np.array([1230, 2780])
sigma = np.array([150, 400])
rv_y_ok = norm.rvs(loc=mu[0], scale=sigma[0], size=1000, random_state=11)
rv_y_nok = norm.rvs(loc=mu[1], scale=sigma[1], size=1000, random_state=0)
y_ok = norm.pdf(x, loc=mu[0], scale=sigma[0])
y_nok = norm.pdf(x, loc=mu[1], scale=sigma[1])

# draw histogram of 2 normal distribution in 3 stages
ax = create_fig()
ax.hist(rv_y_ok,  range=(0,4500), bins=100, density=True, alpha=0.75, color=GREEN)
plt.savefig(path/'normal_hist_s1.png', dpi=300)

ax.hist(rv_y_nok, range=(0,4500), bins=100, density=True, alpha=0.75, color=RED)
plt.savefig(path/'normal_hist_s2.png', dpi=300)

ax.plot(x, y_ok,  c=GREEN, zorder=10, lw=4)
ax.plot(x, y_nok, c=RED, zorder=10, lw=4)
plt.savefig(path/'normal_hist_s3.png', dpi=300)
#%%
# draw PDFs, as well as FP area and FN area based on 2 different thresholds
sigma_rule = 3
ax = create_fig()
ax.plot(x, y_ok,  c=GREEN, zorder=10, lw=2)
ax.plot(x, y_nok, c=RED  , zorder=10, lw=2)
plt.savefig(path/'normal_pdf_s0.png', dpi=300)

th0 = mu[0]+sigma[0]*sigma_rule
th1 = mu[1]-sigma[1]*sigma_rule
ax.axvline(th1, zorder=10, lw=3, color='orange')
ax.fill_between(x[x>th1], 0, y_ok[x>th1], color='orange', alpha=0.7)
plt.savefig(path/'normal_pdf_s1.png', dpi=300)
ax.axvline(th0, zorder=10, lw=3, color='navy')
ax.fill_between(x[x<th0], 0, y_nok[x<th0], color='navy', alpha=0.7)
plt.savefig(path/'normal_pdf_s2.png', dpi=300)

# report error rate
threshold = th0
print(f'宁纵勿枉: threshold = {threshold:d}')
print(f'P(X_OK >{threshold:d}), {1/norm.sf(threshold, loc=mu[0], scale=sigma[0]):.0f}个合格品约有一件会被误判为不合格品')
print(f'P(X_NOK<{threshold:d}), {1/(1-norm.sf(threshold, loc=mu[1], scale=sigma[1])):.0f}个不合格品约有一件会被误判为合格品')
threshold = th1
print(f'宁枉勿纵: threshold = {threshold:d}')
print(f'P(X_OK >{threshold:d}), {1/norm.sf(threshold, loc=mu[0], scale=sigma[0]):.0f}个合格品约有一件会被误判为不合格品')
print(f'P(X_NOK<{threshold:d}), {1/(1-norm.sf(threshold, loc=mu[1], scale=sigma[1])):.0f}个不合格品约有一件会被误判为合格品')

th2 = mu[1]-sigma[1]*3.5
th3 = mu[0]+sigma[0]*6
ax = create_fig()
ax.plot(x, y_ok,  c=GREEN, zorder=10, lw=2)
ax.plot(x, y_nok, c=RED  , zorder=10, lw=2)

ax.axvline(th3, zorder=10, lw=3, color='navy')
ax.fill_between(x[x<th3], 0, y_nok[x<th3], color='navy', alpha=0.7)
plt.savefig(path/'normal_pdf_s3.png', dpi=300)
ax.axvline(th2, zorder=10, lw=3, color='orange')
ax.fill_between(x[x>th2], 0, y_ok[x>th2], color='orange', alpha=0.7)
plt.savefig(path/'normal_pdf_s4.png', dpi=300)
#%%
# draw PDFs, as well as FN area based on specific thresholds
ax = create_fig()
ax.plot(x, y_ok,  c=GREEN, zorder=10, lw=2)
ax.plot(x, y_nok, c=RED  , zorder=10, lw=2)
th_val = mu[0]+sigma[0]
print(th_val)
ax.axvline(th_val, zorder=10, lw=3, color='orange')
ax.fill_between(x[x>th_val], 0, y_ok[x>th_val], color='orange', alpha=0.7)
plt.savefig(path/'normal_cdf.png', dpi=300)

# %%
# * draw PDFs with different paremeters, and draw a thresholds
y_nok = norm.pdf(x, loc=3285, scale=441)
y_ok = norm.pdf(x, loc=760, scale=200)

ax = create_fig()
ax.plot(x, y_ok,  c=GREEN, zorder=10, lw=2)
ax.plot(x, y_nok, c=RED  , zorder=10, lw=2)
plt.savefig(path/'normal_pdf_s5.png', dpi=300)
ax.axvline(1820, zorder=10, lw=3, color='navy')
plt.savefig(path/'normal_pdf_s6.png', dpi=300)
# %%
