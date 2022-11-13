# %%
from pathlib import Path
PATH = Path('moment_estimation/')
PATH.mkdir(exist_ok=True)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 40
plt.rcParams["ytick.labelsize"] = 40
def gen_marker(fname:str, rotation:float=180):
    """Generate maker from svg image file.
    Args:
        fname (str): filename of svg image.
        rotation (int, optional): 
            degree of rotation of original images. Defaults to 180.
    Returns:
        Object of marker.
    """
    person_path, attributes = svg2paths(fname)
    person_marker = parse_path(attributes[0]['d'])
    person_marker.vertices -= person_marker.vertices.mean(axis=0)
    person_marker = person_marker.transformed(mpl.transforms.Affine2D().rotate_deg(rotation))
    person_marker = person_marker.transformed(mpl.transforms.Affine2D().scale(-1,1))
    return person_marker

car_marker = gen_marker('icons/car.svg',)

xx, yy = np.meshgrid(np.arange(4), np.arange(4))
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.scatter(xx.flatten(),yy.flatten(),  s=5000, color='k', marker=car_marker)
ax.axis('off')

# %%
#数据
np.random.seed(1901)
mean,std=650,10 #均值，标准差
size=10000
data1=np.random.normal(mean,std,int(size*1))
#print(data1.shape,data1_new.shape)
data1_norm=(data1-data1.min())/(data1.max()-data1.min())

n = 10
data=data1.reshape(-1,n)

data_mean=np.mean(data,axis=1)
data_sigma=np.mean(data**2, axis=1) - data_mean**2
index=np.arange(1,1001,1)

# %%
from scipy.optimize import fsolve

def newton(m3,m4,xl=-10,xu=10,x0=None):
    if (x0==None):
        x0 = (xl+xu)/2  # init for Newton iteration
    f = lambda x: 2 * x**6 - 4*m3 * x**3 + 3*m4 * x**2 - m3**2
    return fsolve(f, x0)


def get_mu(data1):
    data1_norm=(data1-min(data1))/(max(data1)-min(data1))*10
    m3 = np.mean(data1_norm**3)
    m4 = np.mean(data1_norm**4)
    mu = newton(m3,m4) *(max(data1)-min(data1))/10+min(data1)
    sigma2 = (((data1**3).mean()-mu**3)/(3*mu))
    return mu, sigma2

data_mean2 = np.zeros([len(data)])
data_sigma2 = np.zeros([len(data)])
for i in range(len(data)):
    data_mean2[i], data_sigma2[i] = get_mu(data[i])

# %%
print(np.mean(data_mean2))
print(np.mean(data_sigma2))
#%%
data_sigma2 /= 10

# %%
class UpdateDist:
    def __init__(self, ax_scatter, ax, ax_colorbar, data0, data1):
        self.cummean = np.cumsum(data0)/np.arange(1, data0.shape[0]+1)
        self.sigma = np.sqrt(np.cumsum(data0**2)/np.arange(1, data0.shape[0]+1) - self.cummean**2)
        self.cummean2 = np.cumsum(data1)/np.arange(1, data1.shape[0]+1)
        self.sigma2 = np.sqrt(np.cumsum(data1**2)/np.arange(1, data1.shape[0]+1) - self.cummean2**2) 
        self.data0 = data0
        self.data1 = data1
        print(self.sigma[-1],self.sigma2[-1])
        xn, yn = 10, 1
        xx, yy = np.meshgrid(np.arange(xn), np.arange(yn))
        self.sc_car  = ax_scatter.scatter(xx.flatten(),yy.flatten(),s=12000, fc=[1,1,1,1], ec=[0,0,0,1], marker=car_marker)
        self.color = np.tile([0,32./255,96./255,1],(int(xn*yn),1))
        ax_scatter.set_xlim(-1,xn)
        ax_scatter.set_ylim(-1,yn)
        ax_scatter.invert_yaxis()
        self.ax_scatter = ax_scatter
        
        # scatter plot:
        self.line_left, = ax[0].plot([], [], 'o',
            color='b',
            markersize=8,
            markerfacecolor='none',
            markeredgewidth=3)
        ax[0].plot([0,200],[100,100],ls='--',color='grey',lw=3, alpha=0.7)
        self.line1 =ax[0].plot([],[],ls='-',color='r',lw=3)[0] 
        self.line11 =ax[0].plot([],[],ls='-',color='#1F77B4',lw=4)[0] 
        self.line12 =ax[0].plot([],[],ls='-',color='#1F77B4',lw=4)[0] 

        # now determine nice limits by hand:
        ymax = np.max(np.fabs(data0))
        ymin = np.min(np.fabs(data0))
        ymax1 = np.max(np.fabs(data1))
        ymin1 = np.min(np.fabs(data1))
        xlim = (0, 200)
        ylim = (int(ymin*0.99), int(ymax*1.01))
        # ylim = (620, 670)
        ax[0].set_title('低阶矩估计',fontsize=50)
        ax[0].set_xlim(xlim)
        ax[0].set_ylim(ylim)
        self.ax = ax 
        
        # scatter plot:
        self.line_right, = ax[1].plot([], [], 'o',
            color='b',
            markersize=8,
            markerfacecolor='none',
            markeredgewidth=3)
        ax[1].plot([0,200],[100,100],ls='--',color='grey',lw=3, alpha=0.7)
        self.line2, =ax[1].plot([],[],ls='-',color='r',lw=3)
        self.line21,=ax[1].plot([],[],ls='-', color='#1F77B4', lw=4)
        self.line22,=ax[1].plot([],[],ls='-', color='#1F77B4',lw=4)
        
        ylim = (int(ymin1*0.99), int(ymax1*1.01))
        # now determine nice limits by hand:
        ax[1].set_title('高阶矩估计',fontsize=50)
        ax[1].set_xlim(xlim)
        ax[1].set_ylim(ylim)
        ax[1].ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
        
        
        #draw colorbar
        self.cm = plt.cm.RdYlBu_r
        gradient = np.atleast_2d(np.linspace(0, 1, 256))
        ax_colorbar.imshow(gradient, aspect='auto', cmap=self.cm, alpha=0.7)
        ax_colorbar.set_yticks([])
        ax_colorbar.set_xticks([0, 255])
        ax_colorbar.set_xticklabels(['低', '高'])
        for size in ax_colorbar.get_xticklabels(): 
            size.set_fontsize('50')
        ax_colorbar.set_title('里程数', fontsize=50)


    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            n_inc = 1
            # update lines
            idx = (i)*n_inc
            self.line_left.set_data(index[:idx], self.data0[:idx])
            self.line_right.set_data(index[:idx], self.data1[:idx])
            self.line1.set_data(index[:idx],self.cummean[:idx])
            self.line11.set_data(index[:idx],self.cummean[:idx]+self.sigma[:idx])
            self.line12.set_data(index[:idx],self.cummean[:idx]-self.sigma[:idx]) 
            
            if (i>1):
                self.shade1.remove()
                self.shade2.remove()
            self.shade1 = self.ax[0].fill_between(
                index[:idx],self.cummean[:idx]-self.sigma[:idx],
                self.cummean[:idx]+self.sigma[:idx],
                fc='#1F77B4', alpha=0.1)
            self.line2.set_data(index[:idx],self.cummean2[:idx])
            self.line21.set_data(index[:idx],self.cummean2[:idx]+self.sigma2[:idx])
            self.line22.set_data(index[:idx],self.cummean2[:idx]-self.sigma2[:idx]) 
            self.shade2 = self.ax[1].fill_between(
                index[:idx],self.cummean2[:idx]-self.sigma2[:idx],
                self.cummean2[:idx]+self.sigma2[:idx],
                fc='#1F77B4', alpha=0.1)
            # update scatter facecolor
            for j in range(10):
                indx=10*i+j
                self.color[j,:] = self.cm(data1_norm[indx])
            self.sc_car.set_facecolor(self.color)
            
        return [self.line_left,]

# %%
fig = plt.figure(figsize=(30,14))
spec1 = plt.GridSpec(ncols=1, nrows=1, left=0.02, right=0.75, top=0.95, bottom=0.73, )
ax0 = fig.add_subplot(spec1[0])
ax0.axis('off')
spec3 = plt.GridSpec(ncols=1, nrows=1, left=0.75, right=0.93, top=0.85, bottom=0.8, )
ax2 = fig.add_subplot(spec3[0])
gs = plt.GridSpec(ncols=2, nrows=1, left=0.10, right=0.96, top=0.65, bottom=0.1, wspace=0.3)
ax = [fig.add_subplot(gsi) for gsi in gs]
for axi in ax:
    axi.set_xlabel('样本组编号', fontsize=50)
    # axi.set_ylabel(r'$\hat{\mu}$', fontsize=80, ha='center', va='bottom')
    axi.set_ylabel(r'$\widehat{\sigma^2}$', fontsize=80, ha='center', va='bottom')

# ud = UpdateDist(ax0, ax, ax2, data_mean, data_mean2)
ud = UpdateDist(ax0, ax, ax2, data_sigma, data_sigma2)
anim = FuncAnimation(fig, ud, frames=216, blit=True)
anim.save(PATH/'sigma.mp4', fps=12, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

# %%