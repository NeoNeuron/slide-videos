if __name__ == '__main__':
    #%%
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['axes.labelsize'] = 16
    from scipy.stats import binom, bernoulli, boltzmann
    from simulate_central_limit import UpdateHistogram
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
            'ylim':(0,1),
        },
        'median': {
            'idx':6,
            'xlim':(0,20),
            'ylim':(0,0.5),
        },
        'large': {
            'idx':14,
            'xlim':(280,350),
            'ylim':(0,0.1),
        },
    }
    for key, val in pm.items():
        fig, ax = plt.subplots(1,1,dpi=300, gridspec_kw=dict(left=0.15, right=0.95, bottom=0.15))
        uh = UpdateHistogram(ax, attendence, (0,n))
        idx = val['idx']
        uh(idx)
        uh.lines[0].set_alpha(0)
        uh._draw_gauss(idx)
        uh.lines[1].set_color('r')
        ax.set_xlim(*val['xlim'])
        ax.set_ylim(*val['ylim'])
        ax.grid()
        fig.savefig(f'n_{key:s}.pdf')
    # %%