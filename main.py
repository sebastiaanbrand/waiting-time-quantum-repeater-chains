from repeater_chain_analyzer import RepeaterChainCalculator, RepeaterChainSampler
import plotting_tools as plot_tools


if __name__ == '__main__':

    # Set parameters for the SWAP-ONLY protocol,
    # we'll only calculate waiting time for now.
    pgen  = 0.1
    pswap = 0.9
    n     = 3
    trunc = 100

    # Semi-analytical algorithm
    calculator = RepeaterChainCalculator(n=n,
                                         pgen=pgen,
                                         pswap=pswap,
                                         trunc=trunc,
                                         outputfolder='results/example_1/')
    # Calling the following function will calculate all waiting time PMFs for
    # the waiting times T_0, T_1, up to T_n, up to Pr(T_n = trunc).
    # The results are stored in csv files in the given folder.
    calculator.calculate()

    # After the calculation we can pass the calculator object to the plot
    # function. However, for this example we'll first load the stored results
    # from the previous computation.
    calculator2 = RepeaterChainCalculator(n=0, pgen=0, pswap=0, trunc=0)
    calculator2.load_from_folder('results/example_1/')

    # We can plot the computed pmfs by calling the plot function and passing
    # the RepeaterChainCalculator object. We can also select which T_n to put
    # in the plot. In this example T_2 and T_3 will be plotted.
    plot_tools.plot_distributions(outputfolder='results/example_1/plots/',
                                  rca=calculator2,
                                  to_plot='pmf',
                                  level_selection=[2,3])


    # The Monte Carlo algorithm works very similarly.
    sampler = RepeaterChainSampler(n=n,
                                   pgen=pgen,
                                   pswap=pswap,
                                   outputfolder='results/example_2/')
    # One of the differences is that here we don't need to set trunc before
    # running the algorithm, however it does need to be set for the plots.
    sampler.sample(sample_size=25000)

    plot_tools.plot_distributions(outputfolder='results/example_2/plots/',
                                  rca=sampler,
                                  to_plot='pmf',
                                  level_selection=[2,3],
                                  trunc=trunc)

    # NOTE: uncomment this section for Werner parameter example:
    """
    # Fidelity (or strictly the Werner parameters) are automatically calculated
    # when w0 is set:
    f0 = 0.95
    w0 = (4*f0 - 1) / 3
    T_coh = 10
    n = 2
    trunc = 25 # NOTE: the semi-analytical algorithm with fidelity calculation
               # runs in O(trunc^4), so setting this too high results in long
               # runtimes
    calculator = RepeaterChainCalculator(n=n,
                                         pgen=pgen,
                                         pswap=pswap,
                                         trunc=trunc,
                                         w0=w0,
                                         T_coh=T_coh,
                                         outputfolder='results/example_3/')
    calculator.calculate()
    plot_tools.plot_distributions(outputfolder='results/example_3/plots/',
                                  rca=calculator,
                                  to_plot='fid')

    sampler = RepeaterChainSampler(n=n,
                                  pgen=pgen,
                                  pswap=pswap,
                                  w0=w0,
                                  T_coh=T_coh,
                                  outputfolder='results/example_4/')
    sampler.sample(sample_size=25000)
    plot_tools.plot_distributions(outputfolder='results/example_4/plots/',
                                  rca=sampler,
                                  to_plot='fid',
                                  trunc=trunc)
    # For the Monte Carlo algorithm distillation rounds between the levels can
    # be added by setting n_dist in the constructor. Communication time for
    # swap and distillation operations on every level can be included by setting
    # for example comm_time=[1,2,4,16,32], where comm_time[level] is the
    # communication time required on level `level`. Example:
#    sampler = RepeaterChainSampler(n=3,
#                                  pgen=0.1,
#                                  pswap=0.5,
#                                  w0=0.933,
#                                  T_coh=100,
#                                  comm_time=[1,2,4,8],
#                                  n_dist=1,
#                                  outputfolder='results/example_5/')
    """
