# %%
from init import *
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
#%%
RED=np.array([230,0,18,255])/255.0
GREEN=np.array([0,176,80,255])/255.0
RED = '#4141C2'
GREEN = '#FF4684'
class UpdateFigure_geo_prob:
    def __init__(self, data:np.ndarray, l:float, ax:plt.Axes, ax_right):
        """Plot the first frame for the animation.

        Args:
            data (np.ndarray): 1-D array of number of passagers for each days
            ax (plt.Axes): axes of scatter plot
        """

        self.data = data
        self.trials = np.arange(data.shape[0])+1

        # vertical lines:
        self.last_sample = None
        self.current_sample = None
        self.hit = 0
        ax.set_title(
            f'{self.hit:5d}/{N:5d}' + r'$\,\,\,\,\,\,\,\pi$=%10.6f'%(0), 
            fontsize=20, usetex=True)

        ax_right.set_xlim(0, data.shape[0]+1)
        ax_right.axhline(np.pi, color='r', ls='--')
        mask = 1-(np.floor(self.data[:,2]) == np.floor(self.data[:,3])).astype(float)
        self.pi_est = np.cumsum(mask)/self.trials
        self.pi_est = 2*l/self.pi_est
        self.pi_est[np.isinf(self.pi_est)]=0
        self.ax = ax
        self.ax_right = ax_right
        self.l = l
        self.pi_line, = ax_right.plot([],[])


    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        nb = 50
        if i >=0 and i<self.data.shape[0]/nb:
            if self.last_sample is not None:
                [line.set_alpha(0.05) for line in self.last_sample]
            ii = nb*i
            mask = np.floor(self.data[ii:ii+nb,2]) == np.floor(self.data[ii:ii+nb,3])
            # update lines
            self.current_sample = self.ax.plot(
                self.data[ii:ii+nb,0:2].T, self.data[ii:ii+nb,2:4].T, color=RED)
            self.current_sample.extend(
                self.ax.plot(self.data[ii:ii+nb,0:2][~mask,:].T, self.data[ii:ii+nb,2:4][~mask,:].T, color=GREEN))
            if i == 0:
                self.hit = (~mask).sum()
            else:
                self.hit += (~mask).sum()
            
            self.ax.set_title(
                f'{self.hit:5d}/{(i+1)*nb:5d}' + r'$\,\,\,\,\,\,\,\pi$=%10.6f'%(2*self.l*(i+1)*nb/self.hit),
                fontsize=20, usetex=True)
            # update last sample
            self.last_sample = self.current_sample
            self.pi_line.set_data(self.trials[:ii+nb], self.pi_est[:ii+nb])
        else:
            self.current_sample = self.ax.plot([],[])
        return self.current_sample
#%%
fig = plt.figure(figsize=(15,4))
gs = plt.GridSpec(1,1, top=0.9, bottom=0.02, left=0.03, right=0.3)
ax0 = fig.add_subplot(gs[0])
ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)

[ax0.axhline(h, color='k') for h in np.arange(4)]
ax0.axis('scaled')
ax0.set_ylim(-0.5,3.5)
ax0.set_xlim(-1,5)
ax0.set_yticks([])
ax0.set_xticks([])

gs = plt.GridSpec(1,1, top=0.8, bottom=0.2, left=0.36, right=0.95)
ax1 = fig.add_subplot(gs[0])
ax1.set_ylabel(r'$\pi$', fontsize=30, rotation=0, ha='right', va='center', usetex=True)

np.random.seed(202008)
N = 4000
x = np.random.rand(N,3)
x[:,0] *= 4
x[:,1] *= 3
x[:,2] *= np.pi
l = 0.99

dx = np.sin(x[:,2])*l/2
dy = np.cos(x[:,2])*l/2
data = np.vstack((x[:,0]+dx, x[:,0]-dx, x[:,1]+dy, x[:,1]-dy)).T

ud = UpdateFigure_geo_prob(data, l, ax0, ax1)
fig.savefig(path/'test.pdf')
#%%
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=90, blit=True)
# save animation as *.mp4
anim.save(path/'buffen_needle.mp4', fps=6, dpi=300, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

# %%
