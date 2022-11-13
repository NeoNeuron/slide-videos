# %%
from gaussian2d import *
from pathlib import Path
path = Path('./normal_2d/')
path.mkdir(exist_ok=True)
fig, ax1, ax2, ax3 = gen_canvas()
# create a figure updater
nframes=72
ud = UpdateFigure(ax1, ax2, ax3)
ud.cov = np.array([[1,-0.8],[-0.8,1]])
ud.set_target('morph', np.array([[0,1.6],[1.6,0]]),nframes)
# user FuncAnimation to generate frames of animation
plt.savefig(path/'test_gauss.pdf')
anim = FuncAnimation(fig, ud, frames=nframes, blit=True)
# save animation as *.mp4
anim.save(path/'2d_gaussian_morph.mp4', fps=24, dpi=200, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])