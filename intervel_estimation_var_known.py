# %%
from pathlib import Path
path = Path('./intervel_estimation/')
path.mkdir(exist_ok=True)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.stats import bernoulli
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 30
plt.rcParams["ytick.labelsize"] = 30

# %% data
np.random.seed(1901)
mean,std=605,10 #均值，标准差
size=(200, 10)
data=np.random.randn(*size)*std+mean
data_norm=(data-data.min())/(data.max()-data.min())

data_mean = np.mean(data, axis=1)
index=np.arange(1,1001,1)
# %%
length=10**0.5*1.96
test=np.zeros_like(data_mean)
for i in range(len(data_mean)):
    test[i]= data_mean[i]-length<=mean<=data_mean[i]+length
test=test.astype('bool')

# %%
test_num=test.astype('int')
frequency=np.zeros_like(data_mean)
for i in range(len(test_num)):
    frequency[i]= sum(test_num[:i+1]) / (i+1)

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
    def __init__(self, ax0, ax_main,ax_colorbar,ax_bottom1,ax_bottom2):

        self.ax0 = ax0
        xn, yn = 10, 1
        xx, yy = np.meshgrid(np.arange(xn), np.arange(yn))
        self.sc_car  = ax0.scatter(xx.flatten(),yy.flatten(),s=4500, 
                                   fc='w', ec='k', marker=car_marker)
        self.color = np.tile([0,32./255,96./255,1],(int(xn*yn),1))
        self.ax0.set_xlim(-1,xn)
        self.ax0.set_ylim(-1,yn)
        self.ax0.invert_yaxis()
        
        # main plot:
        self.line_main1, = ax_main.plot([], [], '--',lw=2,color='k')
        self.line_main2, = ax_main.plot([], [], '--',lw=2,color='k')
        ax_main.axvline(x=mean,linestyle='--',color='r',lw=5)
        ax_main.text(mean+0.3,-0.1,r'$\mu$',fontsize=50, va='top', ha='left')
        
        # 阴影部分
        self.shadow=ax_main.fill_between([0,0],1,1,fc='#B4C7E7', alpha=0.3)

        # now determine nice limits by hand:
        xlim = (min(data_mean)-length-1, max(data_mean)+length+1)
        #ax_main.set_title('区间长度与统计量选取的关系',fontsize=40)
        ax_main.set_xlim(xlim)
        ax_main.axhline(0, color='#2F5597',lw=5) 
        ax_main.set_ylim((-1,1))
        ax_main.set_yticks(())
        self.ax_main = ax_main 
        
        # bottom plot1:
        self.rects = ax_bottom1.barh([1,2], [0,0], ) #条形图
        for rec, color in zip(self.rects, ('#B4C7E7', '#2F5597')):# [0,176./255,80./255,1],[228./255,131./255,18./255,1] )):
            rec.set_color(color)
                 
        self.bottom1 = ax_bottom1
        self.bottom1.set_yticks([1,2])
        self.bottom1.set_yticklabels(["", ""])
        self.bottom1.set_xlabel("置信区间数", fontsize=40)
        self.bottom1.text(-21, 1+0.18, r'$\mu$在', fontsize=30, va='center', ha='center', color='k', )
        self.bottom1.text(-21, 1-0.18, "置信区间内", fontsize=30, va='center', ha='center', color='k', )
        self.bottom1.text(-21, 2+0.18, r"$\mu$不在", fontsize=30, va='center', ha='center', color='k', )
        self.bottom1.text(-21, 2-0.18, "置信区间内", fontsize=30, va='center', ha='center', color='k', )
        # Set up plot parameters
        self.bottom1.set_ylim(0.4, 2.6)
        self.bottom1.set_xlim(0, 200)
        self.bottom1.spines['top'].set_visible(False)
        self.bottom1.spines['right'].set_visible(False)
        
        #bottom plot2
        self.line2,=ax_bottom2.plot([],[],linestyle='-',color='r',lw=5,zorder=1)
        ax_bottom2.set_xlabel('置信区间数', fontsize=40)
        ax_bottom2.set_ylabel('在置信区间内的累计频率', fontsize=40)
        ax_bottom2.plot([0,200],[0.95,0.95],'--',color='grey',lw=5,zorder=0)
        ax_bottom2.text(200,0.93,r'$\alpha=0.95$',fontsize=40,color='grey',zorder=0, ha='right', va='top')
        
        # now determine nice limits by hand:
        xlim1 = (-1, frequency.shape[0]+1)
        ylim1 = (0.5, 1)
        #ax_right.set_title('区间长度与样本数n的关系',fontsize=40)
        ax_bottom2.set_xlim(xlim1)
        ax_bottom2.set_ylim(ylim1)
        self.ax_bottom2 = ax_bottom2
        
        #draw colorbar
        self.cm = plt.cm.Blues
        gradient = np.atleast_2d(np.linspace(0, 1, 256))
        ax_colorbar.imshow(gradient, aspect='auto', cmap=self.cm, alpha=0.7)
        ax_colorbar.set_yticks([])
        ax_colorbar.set_xticks([0, 255])
        ax_colorbar.set_xticklabels(['560', '630'])
        ax_colorbar.set_title('里程数', fontsize=30)


    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i <= data_mean.shape[0]:
            if i == 0:
                i = 20 # frontpage
            n_inc = 1
            # update lines
            idx = (i)*n_inc
            #更新区间图
            self.line_main1.set_data([data_mean[i]-length,data_mean[i]-length],[-0.5,0.5])
            self.line_main2.set_data([data_mean[i]+length,data_mean[i]+length],[-0.5,0.5])
            self.shadow.set_verts([[[data_mean[i]-length,-0.5],[data_mean[i]-length,0.5],[data_mean[i]+length,0.5],[data_mean[i]+length,-0.5]]])
            #更新直方图
            negative = np.sum(~test[:i+1]) #未索赔车辆数
            positive = np.sum(test[:i+1]) #索赔车辆数
            for rect, h in zip(self.rects, [positive, negative]): 
                rect.set_width(h)
            #更新频率图
            self.line2.set_data(index[:idx], frequency[:idx])
            
            # update scatter facecolor
            self.color = self.cm(data_norm[idx-1])
            self.sc_car.set_facecolor(self.color)
            
        return self.rects
    
    
fig = plt.figure(figsize=(30,12))
spec1 = gridspec.GridSpec(ncols=1, nrows=1, left=0.06, right=0.43, top=1.0, bottom=0.8, figure=fig)
ax0 = fig.add_subplot(spec1[0])
ax0.axis('off')
spec3 = gridspec.GridSpec(ncols=1, nrows=1, left=0.43, right=0.52, top=0.92, bottom=0.89, figure=fig)
ax2 = fig.add_subplot(spec3[0]) 
spec4 = gridspec.GridSpec(ncols=1, nrows=1, left=0.08, right=0.52, top=0.80, bottom=0.50, figure=fig)
ax4=fig.add_subplot(spec4[0]) 

spec2 = gridspec.GridSpec(ncols=1, nrows=1, left=0.08, right=0.52, top=0.40, bottom=0.1, wspace=0.25, figure=fig)
ax1 = fig.add_subplot(spec2[0])
spec2 = gridspec.GridSpec(ncols=1, nrows=1, left=0.58, right=0.95, top=0.95, bottom=0.1, wspace=0.25, figure=fig)
ax3 = fig.add_subplot(spec2[0])
ud = UpdateDist(ax0, ax4, ax2,ax1,ax3)
plt.savefig(path/'test.pdf')
#%%
anim = FuncAnimation(fig, ud, frames=198, blit=True)
anim.save(path/'mileagejie_2.mp4', fps=10, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%



