# %%
from init import *
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 40
plt.rcParams["ytick.labelsize"] = 40

# %%
np.random.seed(2023)
data = np.random.poisson(10, 10000)
class UpdateDist:
    def __init__(self,ax1,ax2,ax3):
        self.colors = dict(
            blue        = '#375492',
            green       = '#88E685',
            dark_green  = '#00683B',
            red         = '#93391E',
            pink        = '#E374B7',
            purple      = '#A268B4',
            black       = '#000000',
        )
        self.color_ = ['blue','green','pink','purple','dark_green']
        self.tex = ax1.text(0.5, 1.12, r'$z=10$', ha='center', va='center',
            color=self.colors['blue'], fontsize=50, transform=ax1.transAxes)
        self.color_id = [self.colors['blue'],self.colors['red']]
        self.line1, = ax1.plot([], [], 'o', ms=15)
        self.line2, = ax2.plot([], [], 'o', ms=15)
        self.line3, = ax3.plot([], [], 'o', color='b', ms=15)
        self.n = 20
        self.data = data.reshape(-1,self.n)
        ax1.set_xlim([-1,self.n])
        ax2.set_xlim([-1,self.n])
        ax1.set_xticks([])
        ax2.set_xticks([i*10 for i in range(self.n//10+1)])
        ax1.set_ylabel(r'订单数($X$)',fontsize=50)
        ax2.set_ylabel(r'利润($Y$)',fontsize=50)
        ax2.set_xlabel('模拟天数(n)',fontsize=50)
        ax1.set_ylim([-1,20])
        ax1.set_yticks([0,10,20])
        ax2.set_ylim([-1,12000])
        ax2.set_yticks([0,5000,10000])
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3

        self.z = [10,20,40,60,80]
        self.meany, = self.ax2.plot([-1,100],[-1000,-1000],ls='--', color= self.colors['red'],linewidth=8)

        ax3.set_xlabel(r'库存大小(z)',fontsize=50)
        ax3.set_xlim([-1,100])
        ax3.set_xticks([0,50,100])
        ax3.set_ylim([-1,10000])
        ax3.set_yticks([0,5000,10000])
        ax3.set_ylabel(r'平均利润($Y_\mathrm{n}$)',fontsize=50)
       
    def __call__(self, i):
        if i == 0:
            i = 20
        if i>0:
            batch = self.n + 5
            idx = i%batch
            if (idx==0 or idx>self.n):
                if (idx==0):
                    self.meany.set_ydata(np.ones(2)*(-1000))
                return [self.line1,]
            if (i==1):
                self.meany.set_ydata(np.ones(2)*(-1000))
                for line in self.ax3.get_lines():
                    line.remove()
                for obj in self.ax3.collections:
                    obj.remove()
        
            id = i//batch
            z = self.z[id]
            color_id = self.colors[self.color_[id]]#self.color_id[id%2] 
            self.tex.set_text(r'$z=%d$'%(z))

            x = self.data[id][:idx] 
            y = x.copy()
            _z = (x>z)
            z_ = (x<=z)
            y[_z] = 10 * z
            y[z_] = 11 * x[z_] - z
            y = y*100
            
            self.line1.set_data(np.arange(idx),x)
            self.line1.set_color(color_id)
            self.line2.set_data(np.arange(idx),y)
            self.line2.set_color(color_id)
            if(idx==self.n):
                y_mean = np.mean(y)
                self.line3 = self.ax3.scatter(self.z[id], [y_mean], #'o',
                    color=color_id,s = 600)
                #self.line3.set_data(self.z[:len(self.y_mean)],self.y_mean)
                self.meany.set_ydata(np.ones(2)*y_mean)
                self.ax3.axvline(z,ymax=y_mean/10000,ls='--', color=color_id, linewidth=6)
        return [self.line1,]
#%%

fig = plt.figure(figsize=(30,14))
spec1 = fig.add_gridspec(ncols=1, nrows=2, left=0.1, right=0.35, top=0.85, bottom=0.12)
ax1 = fig.add_subplot(spec1[0])
ax2 = fig.add_subplot(spec1[1])
spec2 = fig.add_gridspec(ncols=1, nrows=1, left=0.45, right=0.92, top=0.92, bottom=0.12)
ax3 = fig.add_subplot(spec2[0])
ud = UpdateDist(ax1,ax2,ax3)
anim = FuncAnimation(fig, ud, frames=125, blit=True)
anim.save(path/'inventory_simulation.mp4', fps=15, dpi=100, codec='libx264', bitrate=-1,extra_args=['-pix_fmt', 'yuv420p'])

# %%



