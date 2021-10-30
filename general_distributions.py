if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.labelsize'] = 16
    from scipy.stats import poisson, geom, hypergeom, uniform, expon, norm
    from simulate_central_limit import UpdateHistogram
    from matplotlib.animation import FuncAnimation
    from moviepy.editor import *

    my_dist_args = {
        'poisson': {
            'gen': poisson,
            'pm': {
                'mu': 0.6,
            },
            'range': (0,350),
        },
        'geom': {
            'gen': geom,
            'pm': {
                'p': 0.5,
            },
            'range': (0,800),
        },
        'hypergeom': {
            'gen': hypergeom,
            'pm': {
                'M': 20,
                'n': 7,
                'N': 12,
            },
            'range': (0,2000),
        },
        'uniform': {
            'gen': uniform,
            'pm': { },
            'range': (0,300),
        },
        'expon': {
            'gen': expon,
            'pm': { },
            'range': (0,450),
        },
        'norm': {
            'gen': norm,
            'pm': { },
            'range': (-350,350),
        },

    }

    n = 350 # number of accumulated samples
    K = 100000 # number of random tests
    for key, item in my_dist_args.items():
        # generate sampling data
        attendence = item['gen'].rvs(**item['pm'], size=(K,n), random_state=1240)

        # calculate the accumulate mean and variance
        # single_mean, single_var  = my_distribution.stats(**my_dist_args, moments='mv')

        fig, ax = plt.subplots(1,1,dpi=300, gridspec_kw=dict(left=0.15, right=0.95, bottom=0.15))

        uh = UpdateHistogram(ax, attendence, item['range'])
        number_list = [1,2,3,4,5,8,12,18,28,43,65,99,151,230,350]
        uh.set_frame_numbers = number_list
        uh.set_colors = plt.cm.Oranges(0.8*np.arange(len(number_list)/len(number_list)))

        anim = FuncAnimation(fig, uh, frames=16, interval=800, blit=True)
        fname = f"evolving_{key:s}.mp4"
        anim.save(fname, dpi=200, codec='mpeg4')

        video = VideoFileClip(fname, audio=False)
        video = video.subclip(0,video.duration)

        video.to_videofile(fname.split('.')[0]+'_recompressed.mp4', fps=24)