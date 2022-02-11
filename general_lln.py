if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.labelsize'] = 16
    from scipy.stats import poisson, geom, hypergeom, uniform, expon, norm, cauchy
    from simulate_central_limit import UpdateCurve
    from matplotlib.animation import FuncAnimation

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

    n = 10000 # number of accumulated samples
    for key, item in my_dist_args.items():
        # generate sampling data
        attendence = item['gen'].rvs(**item['pm'], size=(n), random_state=1240)
        single_mean = item['gen'].stats(**item['pm'], moments='m')
        cummean = np.cumsum(attendence)/np.arange(1, attendence.shape[0]+1)

        fig, ax = plt.subplots(
            1,1, figsize=(4,3.5), dpi=200, 
            gridspec_kw=dict(left=0.18, right=0.95, bottom=0.24))

        uh = UpdateCurve(ax, cummean, autolim=True, )
        ax.axhline(single_mean, ls='--', color='r')
        uh.ax.set_title(r'$n$ = '+uh.ax.get_title()[-5:], fontsize=20)
        if 'xlim' in item:
            uh.ax.set_xlim(*item['xlim'])

        factor = np.arange(20)
        for i in range(8):
            if i == 0:
                n_in_each_frame = factor.copy()
            else:
                n_in_each_frame = np.append(n_in_each_frame, factor*10*(i+1)+n_in_each_frame[-1])
        n_in_each_frame = n_in_each_frame[:-2]
        uh.set_frame_numbers(n_in_each_frame+1)

        anim = FuncAnimation(fig, uh, frames=n_in_each_frame.shape[0], blit=True)
        fname = f"evolving_{key:s}.mp4"
        anim.save(fname, fps=10, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])