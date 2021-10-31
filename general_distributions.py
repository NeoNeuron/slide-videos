if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.labelsize'] = 16
    from scipy.stats import poisson, geom, hypergeom, uniform, expon, norm, cauchy
    from simulate_central_limit import UpdateHistogram, Gaussian
    from matplotlib.animation import FuncAnimation
    from moviepy.editor import *

    theme_color = np.array([
        [73,148,196],
        [65,130,164],
        [50,120,138],
        [0,109,135],
    ])

    my_dist_args = {
        'poisson': {
            'gen': poisson,
            'pm': {
                'mu': 0.6,
            },
            'range': (-350,350),
            'ylim': (0, 0.6),
        },
        'geom': {
            'gen': geom,
            'pm': {
                'p': 0.5,
            },
            'range': (-800,800),
            'ylim': (0, 0.6),
        },
        'hypergeom': {
            'gen': hypergeom,
            'pm': {
                'M': 20,
                'n': 7,
                'N': 12,
            },
            'range': (-2000,2000),
            'ylim': (0, 0.4),
        },
        'uniform': {
            'gen': uniform,
            'pm': { },
            'range': (-300,300),
            'ylim': (0, 0.6),
            # 'xlim': (-5,5),
        },
        'expon': {
            'gen': expon,
            'pm': { },
            'range': (-450,450),
            'ylim': (0, 0.4),
        },
        'norm': {
            'gen': norm,
            'pm': { },
            'range': (-350,350),
            'ylim': (0, 0.4),
        },
        # 'cauchy': {
        #     'gen': cauchy,
        #     'pm': { },
        #     'range': (-350,350),
        #     'ylim': (0, 0.4),
        #     # 'xlim': (-10,10),
        # },

    }

    n = 350 # number of accumulated samples
    K = 100000 # number of random tests
    zscore=False
    for key, item in my_dist_args.items():
        # generate sampling data
        attendence = item['gen'].rvs(**item['pm'], size=(K,n), random_state=1240)
        # for i in range(attendence.shape[1]):
        #     attendence[:, i] = attendence[:,0]

        fig, ax = plt.subplots(
            1,1, figsize=(4,3.5), dpi=200, 
            gridspec_kw=dict(left=0.18, right=0.95, bottom=0.24))

        uh = UpdateHistogram(
            ax, attendence, item['range'], 
            zscore=zscore, autolim=not zscore, 
            fade=zscore, envelope_curve='joint')
        uh.ax.set_ylim(*item['ylim'])
        uh.ax.set_ylabel('概率密度')
        if zscore:
            uh.ax.set_xlabel(r'$\frac{1}{\sigma/\sqrt{n}}\left(\frac{1}{n}\sum_i^n X_i-\mu\right)$', fontsize=14)
            x_grid = np.linspace(-10,10,400)
            normal_curve = Gaussian(0,1)(x_grid)/(x_grid[1]-x_grid[0])
            uh.ax.plot(x_grid, normal_curve, 'r')
        else:
            uh.ax.set_xlabel(r'$\sum_i^n X_i$', fontsize=14)
        uh.ax.set_title(r'$n$ : '+uh.ax.get_title()[-5:], fontsize=20)
        if 'xlim' in item:
            uh.ax.set_xlim(*item['xlim'])

        number_list = [1,2,3,4,5,8,12,18,28,43,65,99,151,230,350]
        uh.set_frame_numbers = number_list
        uh.set_colors = plt.cm.Oranges(0.8*np.arange(len(number_list)/len(number_list)))

        anim = FuncAnimation(fig, uh, frames=16, interval=800, blit=True)
        fname = f"evolving_{key:s}.mp4"
        anim.save(fname, dpi=200, codec='mpeg4')

        video = VideoFileClip(fname, audio=False)
        video = video.subclip(0,video.duration)

        video.to_videofile(fname.split('.')[0]+'_recompressed.mp4', fps=24)