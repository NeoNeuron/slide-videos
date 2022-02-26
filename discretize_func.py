# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.animation import FuncAnimation
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.left"] = False
#%%
class UpdateFigure:
    def __init__(self, 
        ax:plt.Axes, data:np.ndarray):
        """Plot the first frame for the animation.

        Args:
            ax (plt.Axes): axes of flight icons.
            data (np.ndarray): 1-D array of number of passagers for each days
        """

        self.colors = dict(
            flight_init=[0,0,0,1],
            main=np.array([0,109,135,255])/255.0, #006D87
            gauss=np.array([177,90,67,255])/255.0, #B15A43
            flight_red=np.array([230,0,18,255])/255.0,
            flight_green=np.array([0,176,80,255])/255.0,
        )
        self.ax = ax
        self.ax.set_xlim(-2,4)
        self.ax.set_ylim(0,0.5)
        self.ax.set_yticks([])
        self.data = data
        self.line=self.ax.plot([],[])

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0 and i < self.data.shape[0]:
            # update lines
            self.ax.cla()
            self.ax.set_xlim(-2,4)
            self.ax.set_ylim(0,0.5)
            self.ax.set_yticks([])
            for xval, yval in zip(np.linspace(-2,4, self.data[i]), norm.pdf(np.linspace(-2,4,self.data[i]),1,1)):
                self.ax.axvline(xval,ymax=yval/0.5, ls='--', lw=1, color='gray')
                line, = self.ax.plot(xval, yval, 'o', color=plt.cm.rainbow(1-(xval+2)/6), ms=10)
                line.set_clip_on(False)
        elif i == self.data.shape[0]:
            self.ax.cla()
            self.ax.set_xlim(-2,4)
            self.ax.set_ylim(0,0.5)
            self.ax.set_yticks([])
            N = 1000
            xvals, yvals = np.linspace(-2,4, N), norm.pdf(np.linspace(-2,4,N),1,1)
            for i in range(N-1):
                line, = self.ax.plot(xvals[i:i+2], yvals[i:i+2], color=plt.cm.rainbow(1-(xvals[i]+2)/6), lw=4)
                line.set_clip_on(False)
        return self.line
# %%
fig, ax = plt.subplots(1,1, figsize=(8,6),dpi=400)

n_segments = 2**np.arange(1,8, dtype=int)
# create a figure updater
ud = UpdateFigure(ax, n_segments)
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=8, blit=True)
# save animation as *.mp4
anim.save('discrete_func.mp4', fps=1, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
# reprocess the video to fix the imcompatible issue for windows user
from moviepy.editor import *

fname = "discrete_func.mp4"
video = VideoFileClip(fname, audio=False)
video = video.subclip(0,video.duration)

video.to_videofile(fname.split('.')[0]+'_cliped.mp4',fps=24)
# %%
