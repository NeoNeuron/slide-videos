if __name__ == '__main__':
    #%%
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.labelsize'] = 16
    from scipy.stats import binom, bernoulli, boltzmann
    from simulate_central_limit import UpdateHistogram, Gaussian
    #%%
    # Simulate Bernoulli random tests

    # my_distribution = boltzmann
    # my_dist_args = dict(
    #     lambda_=1.4,
    #     N = 19,
    # )
    my_distribution = bernoulli
    my_dist_args = dict(
        p=0.9,
    )

    n = 350 # number of accumulated samples
    K = 100000 # number of random tests

    # calculate the accumulate mean and variance
    # single_mean, single_var  = my_distribution.stats(**my_dist_args, moments='mv')
    # generate sampling data
    attendence = my_distribution.rvs(**my_dist_args, size=(K,n), random_state=1240)

    # %%
    # draw static plots
    pm = {
        'small': {
            'idx':0,
            'xlim':(-1,3),
            'ylim':(0,1.4),
        },
        'median': {
            'idx':6,
            'xlim':(0,20),
            'ylim':(0,0.45),
        },
        'large': {
            'idx':14,
            'xlim':(280,350),
            'ylim':(0,0.09),
        },
    }
    for key, val in pm.items():
        fig, ax = plt.subplots(
            1,1, figsize=(4,3), dpi=200, 
            gridspec_kw=dict(left=0.16, right=0.95, bottom=0.16))

        uh = UpdateHistogram(
            ax, attendence, (-n,n), 
            zscore=False, autolim=False, 
            fade=False, envelope_curve=False)
        idx = val['idx']
        uh(idx)
        # draw a smooth gaussian curve
        x_grid = np.linspace(*val['xlim'],400)
        p = 0.9
        nn = uh.number_of_sample_list[idx]
        normal_curve = Gaussian(nn*p, nn*p*(1-p))(x_grid)/(x_grid[1]-x_grid[0])
        uh.ax.plot(x_grid, normal_curve, 'r--')

        ax.set_xlim(*val['xlim'])
        ax.set_ylim(*val['ylim'])
        ax.grid()
        fig.savefig(f'n_{key:s}.pdf')
    # %%