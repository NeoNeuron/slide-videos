if __name__ == '__main__':
    #%%
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.labelsize'] = 16
    from scipy.stats import bernoulli, uniform, norm
    from simulate_central_limit import UpdateHistogram, Gaussian
    from matplotlib.animation import FuncAnimation

    ticket_price = 1
    my_dist_args = {
        'bernoulli': {
            'gen': bernoulli,
            'pm': {
                'p':0.9,
             },
            'range': (-350,350),
            'ylim': (0, 0.4),
        },
        'uniform': {
            'gen': uniform,
            'pm': {
                'scale':ticket_price
             },
            'range': (0,500),
            'ylim': (0, 0.4),
        },
        'norm': {
            'gen': norm,
            'pm': {
                'loc':0.45*ticket_price,
                'scale':np.sqrt(0.3-0.45**2)*ticket_price
             },
            'range': (-350,350),
            'ylim': (0, 0.6),
        },
    }

    threshold = my_dist_args['norm']['gen'].ppf(0.2, **my_dist_args['norm']['pm'])

    n = 350 # number of accumulated samples
    K = 100000 # number of random tests
    zscore=False
    # for key, item in my_dist_args.items():
    # generate sampling data
    # attendence = key['gen'].rvs(**item['pm'], size=(K,n), random_state=1240)
    uniform_rvs = my_dist_args['uniform']['gen'].rvs(**my_dist_args['uniform']['pm'], size=(K,n), random_state=1240)
    bernoulli_rvs = my_dist_args['bernoulli']['gen'].rvs(**my_dist_args['bernoulli']['pm'], size=(K,n), random_state=12)
    attendence = uniform_rvs*bernoulli_rvs

    fig, ax = plt.subplots(
        1,1, figsize=(4,3.5), dpi=200, 
        gridspec_kw=dict(left=0.18, right=0.95, bottom=0.24))

    uh = UpdateHistogram(
        ax, attendence, (-300,300), 
        zscore=zscore, autolim=not zscore, 
        fade=zscore, envelope_curve='joint', 
        xlabel_scale=0.1)
    uh.ax.set_ylim(my_dist_args['norm']['ylim'])
    uh.ax.set_ylabel('概率密度')
    if zscore:
        uh.ax.set_xlabel(r'$\frac{1}{\sigma/\sqrt{n}}\left(\frac{1}{n}\sum_i^n X_i-\mu\right)$', fontsize=14)
        x_grid = np.linspace(-10,10,400)
        normal_curve = Gaussian(0,1)(x_grid)/(x_grid[1]-x_grid[0])
        uh.ax.plot(x_grid, normal_curve, 'r')
        # uh.ax.set_title(r'$n$ : '+uh.ax.get_title()[-5:], fontsize=20)
    else:
        uh.ax.set_xlabel('总利润(万)', fontsize=14)
        # uh.ax.set_xlabel(r'$\sum_i^n X_i$', fontsize=14)
    # if 'xlim' in item:
    #     uh.ax.set_xlim(*item['xlim'])
    # uh.ax.set_xlim(*(-0.5,250))

    number_list = [1,3,5,6,7,8,12,18,28,43,65,99,151,230,350]
    uh.set_frame_numbers = number_list
    uh.set_colors = plt.cm.Oranges(0.8*np.arange(len(number_list)/len(number_list)))

    anim = FuncAnimation(fig, uh, frames=16, blit=True)
    if zscore:
        fname = f"evolving_joint_norm.mp4"
    else:
        fname = f"evolving_joint.mp4"
    anim.save(fname, fps=1, dpi=100, codec='libx264', bitrate=-1, extra_args=['-pix_fmt', 'yuv420p'])