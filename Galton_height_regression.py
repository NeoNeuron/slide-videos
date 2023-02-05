#%%
import numpy as np
from pathlib import Path
path = Path('./LinearRegression/')
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

gs_font = fm.FontProperties(
                fname='/System/Library/Fonts/Supplemental/GillSans.ttc')

plt.style.use('old-style')

N = 8
X = np.arange(N)[:,None]+65
np.random.seed(0)
Y = np.random.randn(N,1)*.1+0.6*(X-X.mean())+X.mean()

linear_regressor = LinearRegression()
reg = linear_regressor.fit(X, Y)

WIDTH, HEIGHT, DPI = 600, 500, 100
fig, ax = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)

ax.scatter(X, Y, color=[0,0,0,0], edgecolors='k', s=100, zorder=10)
Xpred = np.linspace(65, 72)[:, None]
Ypred = linear_regressor.predict(Xpred)
ax.plot(Xpred, Ypred, 'k')
ax.plot(X, X, color='grey')
ax.axvline(X.mean(), ls='--', color='grey')
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.set_xlabel(' '.join('PARENTS HEIGHT')+ '  (inches)',
              fontproperties=gs_font, fontsize=13)
ax.set_ylabel(' '.join('CHILDREN HEIGHT')+'  (inches)',
              fontproperties=gs_font, fontsize=13)
ax.set_xlim(64.55, 72.6)
ax.set_ylim(64.8, 72.3)
# ax.set_yticks([3.6, 3.8, 4.0, 4.2])
ax.tick_params('x', which='both', top=True, bottom=True)
ax.tick_params('y', which='both', right=True, left=True)
ax.text(0.32, 0.8, 'AVERAGE HEIGHT', 
        transform=ax.transAxes,
        ha='center', va='center', fontsize=13,
        )
ax.text(0.75, 0.85, 'Y=X', color='grey',
        transform=ax.transAxes, fontproperties=gs_font,
        ha='center', va='center', fontsize=17,
        )

for tick in ax.get_xticklabels():
    tick.set_fontname("Gill Sans")
    tick.set_fontsize(14)
for tick in ax.get_yticklabels():
    tick.set_fontname("Gill Sans")
    tick.set_fontsize(14)

plt.tight_layout()
plt.savefig(path/'Galton_height.png', dpi=DPI)
# %%
from functools import partial
from random import gauss, randrange
from PIL import Image, ImageFilter

def add_noise_to_image(im, perc=20, blur=0.5):
    gaussian = partial(gauss, 0.50, 0.02)
    width, height = im.size
    for _ in range(width*height * perc // 100):
        noise = int(gaussian() * 255)
        x, y = randrange(width), randrange(height)
        r, g, b, a = im.getpixel((x, y))
        im.putpixel((x, y),
                    (min(r+noise, 255), min(g+noise, 255), min(b+noise, 255)))

    im = im.filter(ImageFilter.GaussianBlur(blur))
    return im

im = Image.open(path/'Galton_height.png')
im = add_noise_to_image(im, 30)
im.save(path/'Galton_height-noisy.png')
# %%
