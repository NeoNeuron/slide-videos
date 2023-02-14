# %%
from pathlib import Path
path = Path('./intervel_estimation/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.stats import t
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24

# %% data
np.random.seed(1901)
mean,std=605,10 #均值，标准差
size=(200, 10)
data=np.random.randn(*size)*std+mean
data_norm=(data-data.min())/(data.max()-data.min())

interval_len = data.std(1) * (-2)*t.ppf(0.025, df = size[1]-1)/np.sqrt(size[1])

# %%
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

# %%
class UpdateDist:
    def __init__(self, ax0, ax_main,ax_colorbar):
        self.ax0 = ax0
        xn, yn = 10, 1
        xx, yy = np.meshgrid(np.arange(xn), np.arange(yn))
        self.sc_car  = ax0.scatter(xx.flatten(),yy.flatten(),s=7000, 
                                   fc='w', ec='k', marker=car_marker)
        self.color = np.tile([0,32./255,96./255,1],(int(xn*yn),1))
        self.ax0.set_xlim(-1,xn)
        self.ax0.set_ylim(-1,yn)
        self.ax0.invert_yaxis()
        
        # scatter plot:
        self.line_main, = ax_main.plot([], [], 'o',
            color='#2F5597',
            markersize=15,
            markerfacecolor='none',
            mew=4)
        self.line1=ax_main.plot([],[],color='orange',lw=5)[0]
        ax_main.plot([0,200],[12.4,12.4],linestyle='--',color='r',lw=5) #sigma已知
        ax_main.text(3,10.5, r'$\sigma$已知', color='r', fontsize=30)
        ax_main.set_xlabel('样本编号', fontsize=40)
        ax_main.set_ylabel('区间长度', fontsize=40)

        # now determine nice limits by hand:
        ymax = np.max(np.fabs(interval_len))
        ymin = np.min(np.fabs(interval_len))
        xlim = (-1, interval_len.shape[0]+1)
        ylim = (int(ymin*0.9), int(ymax*1.1))
        #ax_main.set_title('区间长度与统计量选取的关系',fontsize=40)
        ax_main.set_xlim(xlim)
        ax_main.set_ylim((2,30))
        self.ax_main = ax_main 
        #TODO: replace with true data
        self.cummean = np.cumsum(interval_len)/np.arange(1, interval_len.shape[0]+1)
        
        #draw colorbar
        self.cm = plt.cm.Blues
        gradient = np.atleast_2d(np.linspace(0, 1, 256))
        ax_colorbar.imshow(gradient, aspect='auto', cmap=self.cm, alpha=0.7)
        ax_colorbar.set_yticks([])
        ax_colorbar.set_xticks([0, 255])
        ax_colorbar.set_xticklabels(['560', '630'])
        ax_colorbar.set_title('里程数', fontsize=35)

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0 and i <= interval_len.shape[0]:
            n_inc = 1
            # update lines
            idx = (i)*n_inc
            xdata = np.arange(idx)
            self.line_main.set_data(xdata, interval_len[:idx])
            self.line1.set_data(xdata,self.cummean[:idx])
            
            # update scatter facecolor
            self.color = self.cm(data_norm[idx-1])
            self.sc_car.set_facecolor(self.color)
            
        if i==201:
            self.ax_main.text(3,self.cummean[-1]+1,r'$\sigma$未知', color='orange', fontsize=30)
            
        return [self.line_main,]
    
# %%
fig = plt.figure(figsize=(20,12),dpi=100)
spec1 = gridspec.GridSpec(ncols=1, nrows=1, left=0.04, right=0.72, top=0.98, bottom=0.8, figure=fig)
ax0 = fig.add_subplot(spec1[0])
ax0.axis('off')
spec3 = gridspec.GridSpec(ncols=1, nrows=1, left=0.72, right=0.9, top=0.92, bottom=0.87, figure=fig)
ax2 = fig.add_subplot(spec3[0])
spec2 = gridspec.GridSpec(ncols=1, nrows=1, left=0.08, right=0.92, top=0.78, bottom=0.1, figure=fig)
ax1 = fig.add_subplot(spec2[0])
ud = UpdateDist(ax0, ax1,ax2)
anim = FuncAnimation(fig, ud, frames=210, blit=True)
#%%
anim.save(path/'mileagejie_1.mp4', fps=10, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%


