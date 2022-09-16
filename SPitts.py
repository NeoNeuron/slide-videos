#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['font.size']=22
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
#%%
k = 1000
np.random.seed(202)
Ns = np.linspace(1, 2e4, 1000, dtype=int)
rewards = np.array([np.mean(2**np.ceil(-np.log2(np.random.rand(k, N))), axis=1) for N in Ns])
# %%
np.save('test.npy', rewards)
# %%
rewards = np.load('test.npy')
fig, ax = plt.subplots(1, 1, figsize=(15,8))
ax.plot(np.tile(Ns, (1000,1)).flatten(), rewards.flatten(), '.', c='g', ms=.1, alpha=0.5)
ax.plot(Ns, (5+np.log10(Ns))/np.log10(2), color='r', label=r'$(5+\log_{10}(N))/\log_{10}(2)$', zorder=10)
ax.plot(Ns, np.percentile(np.array(rewards), 90, axis=1), label='90%分位数', color='orange')
# ax.plot(Ns, np.mean(np.array(rewards), axis=1), label='mean', color='navy')
ax.set_xlabel('游戏次数(N)')
ax.set_ylabel('平均受益(元)')
ax.set_xlim(0,20000)
ax.set_ylim(0,100)
ax.legend(fontsize=20)
fig.savefig('N_reward.png', dpi=200)
# %%
np.random.seed(31032)
N = 3_000
rewards = 2**np.floor(-np.log2(np.random.rand(N)))
cum_reward = np.cumsum(rewards)/np.arange(1,N+1)
Ns = np.arange(1,N+1)

fig, ax = plt.subplots(2,1,figsize=(10,10), sharex=True, gridspec_kw={'hspace':0.1})
# plt.plot(Ns, np.log10(Ns)/np.log10(2), color='r', label=r'$\log_{10}(N)/\log_{10}(2)$')
ax[0].plot(Ns[::1], rewards[::1], c='g', lw=2)[0].set_clip_on(False)
ax[1].plot(Ns[::1], cum_reward[::1], c='g', lw=2)
# plt.plot(Ns, np.percentile(np.array(rewards), 99, axis=1), label='median', color='orange')
# plt.plot(Ns, np.mean(np.array(rewards), axis=1), label='mean', color='navy')
ax[0].set_ylim(0)
ax[0].set_ylabel('单次奖金(元)')
ax[1].set_xlabel('重复游戏次数')
ax[1].set_ylabel('累计平均奖金(元)')
ax[1].set_xlim(0,3000)
[axi.grid() for axi in ax]
fig.savefig('a_player_3k.png', dpi=200)

# %%
class UpdateFigure:
    def __init__(self, ax, data):
        self.ax = ax
        self.data = data
        # self.line, = self.ax.plot([], [], 'o', color='navy', ms=10)
        self.line, = self.ax.plot(np.arange(self.data.shape[0])+1, self.data, 'o', color='royalblue', ms=10, markeredgecolor='orange', alpha=.9)
        self.line.set_clip_on(False)


    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i <= self.data.shape[0]:
            self.line.set_data(np.arange(i)+1, self.data[:i])
        elif i == self.data.shape[0]+1: 
            self.ax.axhline(np.percentile(self.data, 90,), color='red', label='90%分位数')
            ax.legend()
        return [self.line,]
# %%
np.random.seed(31032)
N = 3_000
k = 100
rewards = np.mean(2**np.floor(-np.log2(np.random.rand(N,k))), axis=0)
#%%
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.set_xlabel('玩家序号')
ax.set_ylabel('平均奖金(元)')
ax.set_ylim(0, 183)
ax.set_xlim(-1, 101)
ud = UpdateFigure(ax, rewards)
# %%
anim = FuncAnimation(fig, ud, frames=120, blit=True)
anim.save('spitts.mp4', fps=12, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

# %%
