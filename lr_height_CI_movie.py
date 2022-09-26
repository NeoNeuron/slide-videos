# %%
PATH = 'LinearRegression/'
import os
os.makedirs(PATH, exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
# matplotlib parameters to ensure correctness of Chinese characters 
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'SimHei'] # Chinese font
plt.rcParams['axes.unicode_minus']=False # correct minus sign

plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 30
plt.rcParams["ytick.labelsize"] = 30

from sklearn.linear_model import LinearRegression

# 300 data
np.random.seed(20)
f = [157, 158, 163, 165,167,167,168,169,170, 172]
f = np.array(f)
n = len(f)
data1= (f-155) + np.random.normal(0,2,n) + 150

class UpdateDist:
    def __init__(self, ax1,ax_bottom1,ax_bottom2):
       
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        
        # main plot:
        ax1.set_xlabel('父母平均身高(cm)',fontsize=60)
        ax1.set_ylabel('女孩身高(cm)',fontsize=60)
        ax1.set_ylim([150,225])
        ax1.set_xlim([155,210])
        ax1.set_xticks([155+i*10 for i in range(6)])
        self.line1, = ax1.plot([], [], '--',lw=3,color='#1f77b4')
        self.line2, = ax1.plot([], [], '--',lw=3,color='#1f77b4')
        self.line, = ax1.plot([], [], '-',lw=3,color='#1f77b4')
        ax1.scatter([208],[203],marker='^',s=700,c='r')
        
        # bottom plot1:
        self.rects = ax_bottom1.barh([1,2], [0,0], ) #条形图初始化
        for rec, color in zip(self.rects, ( [0,176./255,80./255,1],[228./255,131./255,18./255,1] )):
            rec.set_color(color)
                 
        self.bottom1 = ax_bottom1
        self.bottom1.set_yticks([1,2])
        self.bottom1.set_yticklabels(["", ""])#ticklabel设空，用这个显示不好看，所以后面设置了text现在
        self.bottom1.set_xlabel("置信区间数", fontsize=60)
        self.bottom1.text(-0.25, 0.34, r'$Y_0$在置', transform=self.bottom1.transAxes, fontsize=50, color=[0,176./255,80./255,1], )
        self.bottom1.text(-0.25, 0.22, "信区间内", transform=self.bottom1.transAxes, fontsize=50, color=[0,176./255,80./255,1], )
        self.bottom1.text(-0.25, 0.69, r"$Y_0$不在置", transform=self.bottom1.transAxes, fontsize=50, color=[228./255,131./255,18./255,1], )
        self.bottom1.text(-0.25, 0.57, "信区间内", transform=self.bottom1.transAxes, fontsize=50, color=[228./255,131./255,18./255,1], )
        # Set up plot parameters
        self.bottom1.set_ylim(0, 3)
        self.bottom1.set_xlim(0, 200)
        self.bottom1.spines['top'].set_visible(False)
        self.bottom1.spines['right'].set_visible(False)
        self.neg = [0]
        self.pos = [0]
        
        
        #bottom plot2
        self.line_b,=ax_bottom2.plot([],[],linestyle='-',color='black',lw=5,zorder=0)
        ax_bottom2.set_xlabel('置信区间数', fontsize=60)
        ax_bottom2.set_ylabel('在置信区间内的累计频率', fontsize=40)
        ax_bottom2.plot([0,200],[0.9,0.9],'--',color='red',lw=5,zorder=0)
        ax_bottom2.text(150,0.83,r'$\alpha=0.9$',fontsize=55,color='b',zorder=1)
        
        # now determine nice limits by hand:
        xlim1 = (-1, 200)
        ylim1 = (0.5, 1)
        #ax_right.set_title('区间长度与样本数n的关系',fontsize=40)
        ax_bottom2.set_xlim(xlim1)
        ax_bottom2.set_ylim(ylim1)
        self.ax_bottom2 = ax_bottom2
        
        

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process

        if i > 0:
            #这边设置一个当前帧的判断函数，判断当前数据的最大频率，来设置ylim
            n_inc = 1
            # update lines
            idx = (i)*n_inc
            #更新区间图
            data1 = (f-155) + np.random.normal(0,2,n) + 150
            if (i==1):
                self.scatter1 = self.ax1.scatter(f,data1,s=200,fc='#1f77b4', ec='navy', )
            else:
                self.scatter1.remove()
                self.scatter1 = self.ax1.scatter(f,data1,s=200,fc='#1f77b4', ec='navy', )
            model = LinearRegression()
            model.fit(f.reshape(-1,1), data1)
            b1 = model.coef_[0]
            b0 = model.intercept_
            X = np.linspace(156,209,100)
            Y = np.zeros([100])
            sigma2 = np.sum((data1-b0-b1*f)**2)/(n-2)
            def get_width(x0):
                t = 1.8595
                return t*np.sqrt(1+1/n + (x0-np.mean(f))**2/np.sum((f-np.mean(f))**2)*sigma2)
            for j in range(100):
                Y[j] = get_width(X[j])

            self.line1.set_data(X, b1*X+b0+Y)
            self.line2.set_data(X, b1*X+b0-Y)
            self.line.set_data(X, b1*X+b0)
            #self.ax_main.fill_between([data_mean[i]-length,data_mean[i]+length],-0.5 ,0.5, fc='b', alpha=0.3)
            #更新直方图
            if_po = (np.abs((208-155)+150 - b0-b1*208)<=get_width(208))
            if if_po:
                self.pos.append(self.pos[-1]+1)
                self.neg.append(self.neg[-1])
            else:
                self.pos.append(self.pos[-1])
                self.neg.append(self.neg[-1]+1)
            for rect, h in zip(self.rects, [self.pos[-1], self.neg[-1]]): 
                rect.set_width(h)#self.rects设置barh中的每一条对象
            #更新频率图
            self.line_b.set_data(np.arange(i), np.array(self.pos[1:])/(np.array(self.pos[1:])+np.array(self.neg[1:])))
            
                
        return self.rects
    
    

fig = plt.figure(figsize=(30,17),dpi=200)

spec1 = gridspec.GridSpec(1, 1, left=0.12, right=0.97, top=0.95, bottom=0.55, figure=fig)
ax1 = fig.add_subplot(spec1[0]) 

spec2 = gridspec.GridSpec(1, 2, left=0.12, right=0.97, top=0.45, bottom=0.1, figure=fig)
ax2 = fig.add_subplot(spec2[0])
ax3 = fig.add_subplot(spec2[1])
ud = UpdateDist(ax1=ax1,ax_bottom1=ax2,ax_bottom2=ax3)
anim = FuncAnimation(fig, ud, frames=198, blit=True)
anim.save(PATH+'temp_xxj_animate_4.mp4', fps=10, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])

# %%
