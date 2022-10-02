# %%
PATH = 'moment_estimation/'
import os
os.makedirs(PATH, exist_ok=True)
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 35
plt.rcParams["ytick.labelsize"] = 35

# %%
class UpdateFigure:
    def __init__(self, ax):
        self.line, =ax.plot([],[],lw=5, color='g')
        self.ax = ax
        self.ax.set_xlim([0,0.14])
        self.ax.set_ylim([0,35])
        self.ax.set_ylabel("最优分组人数k", fontsize=60) 
        self.ax.set_xlabel("感染概率p", fontsize=60)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            self.line, = self.ax.plot([], [], lw=5, color='g')

        if i <= 199:
            # update curve
            xdata, ydata = self.line.get_data()
            p=i*0.14/200
            if len(xdata) == 0:
                xdata = [0]
                ydata = [37]
            else:
                xdata =np.append(xdata, p) 
                ydata =np.append(ydata, math.log(p,0.562)) 
            self.line.set_data(xdata, ydata)
        elif i==200:
            self.ax.plot([0,0.1], [4,4], lw=3, ls="--",color='black')
            self.ax.plot([0.10,0.10], [0,4], lw=3, ls="--",color='black')
            self.ax.text(0.01,0.14,"k=4", transform=self.ax.transAxes, fontsize=35, color='black',)
            self.ax.text(0.73,0.01,"p=0.1", transform=self.ax.transAxes, fontsize=35, color='black',)
        return [self.line]

# %%
fig, ax = plt.subplots(1,1, figsize=(18,8),dpi=200, gridspec_kw=dict(left=0.10, right=0.96, top=0.96, bottom=0.18, wspace=0.3))
ud = UpdateFigure(ax)
anim = FuncAnimation(fig, ud, frames=216, blit=True)
anim.save(PATH+'curve_p.mp4', fps=24, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

# %%
np.log()


