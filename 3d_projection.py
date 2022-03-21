#%%
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
# %%
plt.rcParams['grid.color'] = '#A8BDB7'
plt.rcParams['grid.linestyle'] = '--'

class UpdateFigure:
    def __init__(self, ax,):

        self.colors = dict(
            blue        = '#375492',
            green       = '#88E685',
            dark_green  = '#00683B',
            red         = '#93391E',
            pink        = '#E374B7',
            purple      = '#A268B4',
            black       = '#000000',
        )
        # ====================
        # adjust figure axis settings
        # ====================
        ax.set_xlabel(r'$y=g(x)$', fontsize=20)
        ax.set_ylabel(r'$x$', fontsize=20)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r'$f(y)$', rotation=0, fontsize=20)
        ax.set_xlim(0,1)
        ax.set_xticks([0,0.5,1])
        ax.set_ylim(0,1)
        ax.set_yticks([0,0.5,1])
        ax.set_zlim(0,3)
        ax.set_zticks([0,3])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.invert_xaxis()
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.set_box_aspect((1,1,0.3))

        # ====================
        # Draw functions
        # ====================
        X_grid = np.linspace(0,1, 100)
        Y_grid = np.linspace(0,1, 100)

        self.g_x = lambda x: 1/(1+np.exp(-(x-0.5)/0.1))
        self.f_y = lambda y: norm.pdf(y, loc=0.5, scale=.15)
        X = self.g_x(Y_grid)
        Z = self.f_y(X_grid)
        ax.plot(X, Y_grid, np.zeros_like(X), color=self.colors['blue'], lw=3)
        ax.plot(X_grid, np.ones_like(X_grid), Z, color=self.colors['green'], lw=3)

        # ====================
        # draw points
        # ====================

        yp = 0.4
        self.ddyp = 0.0001
        self.dyp = 0.02

        verts, verts_v, verts_v1 = self.get_verts(yp)

        line1, = ax.plot([self.g_x(yp), self.g_x(yp)], [yp, 1.0], [0,0], 
            ls='--', color=self.colors['black'], alpha=0.3)
        line2, = ax.plot([self.g_x(yp+self.dyp), self.g_x(yp+self.dyp)], [yp+self.dyp, 1.0], [0,0],
            ls='--', color=self.colors['black'], alpha=0.3)
        self.lines = [line1, line2]

        # ax.plot([g_x(yp), g_x(yp)], [1.0, 1.0], [0, f_y(g_x(yp))], ls='--', color='purple', alpha=0.5)
        # ax.plot([g_x(yp+dyp), g_x(yp+dyp)], [1.0, 1.0], [0,f_y(g_x(yp+dyp))], ls='--', color='purple', alpha=0.5)

        self.shade = Poly3DCollection([verts], facecolor=self.colors['blue'], edgecolor='None', alpha=0.4) # Add a polygon instead of fill_between
        self.shade_v = Poly3DCollection([verts_v], facecolor=self.colors['red'], edgecolor='None', alpha=0.4) # Add a polygon instead of fill_between
        self.shade_v1 = Poly3DCollection([verts_v1], facecolor=self.colors['green'], edgecolor='None', alpha=0.4) # Add a polygon instead of fill_between
        ax.add_collection3d(self.shade)
        ax.add_collection3d(self.shade_v)
        ax.add_collection3d(self.shade_v1)
        self.text1 = ax.text(-0.05,yp*0.9,0, r'$\Delta x$', ha='left',
            color=self.colors['blue'], fontsize=16)
        self.text2 = ax.text(self.g_x(yp)*1.2,0.78,0, r"$g'(x)\Delta x$", ha='left', 
            color=self.colors['red'], fontsize=16)
        self.text3 = ax.text(self.g_x(yp)*1.2,1,self.f_y(self.g_x(yp))*1.0, r"$f(g(x))$", ha='left', 
            color=self.colors['dark_green'], fontsize=16)

        self.yp = yp
        self.ax = ax

    def get_verts(self, yp_):
        yp_array = np.arange(yp_, yp_+self.dyp+self.ddyp, self.ddyp)
        verts = [(self.g_x(yi), yi, 0) for yi in yp_array] \
                + [(0, yp_+self.dyp, 0), (0, yp_, 0)]

        xp_array = self.g_x(yp_array)
        verts_v = [(xi, 1, self.f_y(xi)) for xi in xp_array] \
                + [(self.g_x(yp_+self.dyp), 1, 0), (self.g_x(yp_), 1, 0)]

        xp_array_1 = self.g_x(np.arange(yp_+self.dyp, 1+self.ddyp, self.ddyp))
        verts_v1 = [(xi, 1, self.f_y(xi)) for xi in xp_array_1] \
                + [(self.g_x(1), 1, 0), (self.g_x(yp_+self.dyp), 1, 0)]
                
        return verts,verts_v,verts_v1

    def __call__(self, i):
        if i > 0:
            self.yp += self.ddyp*10
            verts, verts_v, verts_v1 = self.get_verts(self.yp)
            self.shade.set_verts([verts])
            self.shade_v.set_verts([verts_v])
            self.shade_v1.set_verts([verts_v1])
            self.lines[0].set_data_3d([self.g_x(self.yp), self.g_x(self.yp)], [self.yp, 1.0],[0,0])
            self.lines[1].set_data_3d([self.g_x(self.yp+self.dyp), self.g_x(self.yp+self.dyp)], [self.yp+self.dyp, 1.0], [0,0])
            self.text1.set_position_3d((-0.05, self.yp*0.9, 0))
            self.text2.set_position_3d((self.g_x(self.yp)*1.2,0.78,0,))
            self.text3.set_position_3d((self.g_x(self.yp)*1.2,1,self.f_y(self.g_x(self.yp))*1.0,))

        return self.lines

fig = plt.figure(figsize=(6,5),dpi=400,)
ax = fig.add_subplot(projection='3d')
# create a figure updater
nframes=180
ud = UpdateFigure(ax,)
#%%
# user FuncAnimation to generate frames of animation
anim = FuncAnimation(fig, ud, frames=nframes+1, blit=True)
# save animation as *.mp4
anim.save('3d_projection_movie.mp4', fps=60, dpi=400, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])
# %%
