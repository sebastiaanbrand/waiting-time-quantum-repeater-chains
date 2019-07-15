from repeater_chain_analyzer import RepeaterChainCalculator, RepeaterChainSampler
import probability_tools as prob_tools

import numpy as np
import matplotlib.pyplot as plt
import pathlib


def plot_distributions(outputfolder, rca, to_plot='pmf', level_selection=None,
                       trunc=None, format='png', verbose=True, show=False):
    """Plots distributions.

    Parameters
    ----------
    outputfolder : string
        Folder where to put the output file.
    rca : RepeaterChainAnalyzer
        RepeaterChainAnalyzer object with the distributions already calculated
        or samples already sampled.
    to_plot : string
        Type of thing to plot. Can be 'pmf', 'cdf', 'fid', or 'wern'.
    level_selection : list of ints
        List of which levels to plot for. If set to None the computed
        distributions for all levels will be plotted.
    trunc : int
        Up to where the x-axis of the plot runs. Cannot be higher than the
        computed t_trunc for the RepeaterChainCalculator. Must be set when rca
        is a RepeaterChainSampler.
    format : string
        Output format (e.g. 'png' or 'pdf').
    verbose : boolean
        If set to True prints stuff to console.
    show : boolean
        If set to True, shows() the plot before saving it.
    """
    outputpath = _init_plot_file(outputfolder, to_plot, format)

    data = _get_data(rca, to_plot, trunc)

    if(level_selection is None):
        level_selection = range(len(data))
    if(trunc is None):
        trunc = len(data[0]) - 1

    if(verbose):
        print("Plotting {} for n = {}".format(to_plot, list(level_selection)))

    # select colors
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    col_counter = 0

    for level, dist in enumerate(data):
        if(level in level_selection):
            col = default_colors[col_counter]
            col_counter += 1

            plt.step(x=range(trunc+1), y=dist[:trunc+1], color=col)

            # plot mean (bounds)
            if(to_plot == 'pmf' or to_plot == 'cdf'):
                if(type(rca) is RepeaterChainCalculator):
                    left, right = rca.mean_bounds(level)
                elif(type(rca) is RepeaterChainSampler):
                    mean, se = rca.sample_mean(level)
                    left  = mean - se
                    right = mean + se
                right = min(int(np.round(right)), trunc)
                left = int(np.round(left))
                if(left <= right):
                    plt.fill_between(x=range(left,right+1),
                                     y1=dist[left:right+1],
                                     color=col, alpha=0.5)

            # DKW conf bands
            if(to_plot == 'cdf' and type(rca) == RepeaterChainSampler):
                n_samps = len(rca.T_samples[level])
                lower, upper = _ecdf_conf_bands(dist, n_samps, alpha=0.01)
                plt.fill_between(x=range(trunc+1), y1=upper, y2=lower,
                                 color=col, alpha=0.2)


    # labels and stuff
    plt.title(_plot_title(rca))
    plt.xlabel("$t$")
    plt.ylabel(_ylab(to_plot))
    plt.ylim(bottom=0)
    legend = []
    for level in level_selection:
        entry = "$N={}$".format(2**level)
        legend.append(entry)
    plt.legend(legend)
    plt.savefig(outputpath, dpi=300, bbox_inches='tight')
    if(show):
        plt.show()
    plt.clf()


def _init_plot_file(outputfolder, to_plot, format):
    """Initializes the plot file. """
    pathlib.Path(outputfolder).mkdir(parents=True, exist_ok=True)
    outputpath = outputfolder + '{}.{}'.format(to_plot, format)
    return outputpath

def _get_data(rca, to_plot, trunc):
    """Returns the relevant data depending on the stuff to plot.

    Parameters
    ----------
    rca : RepeaterChainAnalyzer
        RepeaterChainAnalyzer object with the distributions already calculated
        or samples already sampled.
    to_plot : string
        Type of thing to plot. Can be 'pmf', 'cdf', 'fid', or 'wern'.
    """
    if(type(rca) is RepeaterChainCalculator):
        if(to_plot == 'pmf'):
            return rca.pmf
        elif(to_plot == 'cdf'):
            cdfs = np.zeros(shape=np.shape(rca.pmf))
            for i, pmf in enumerate(rca.pmf):
                cdfs[i] = prob_tools.pmf_to_cdf(pmf)
            return cdfs
        elif(to_plot == 'wern'):
            wern = rca.wern
            wern[rca.pmf == 0] = np.nan
            return wern
        elif(to_plot == 'fid'):
            wern = rca.wern
            wern[rca.pmf == 0] = np.nan
            return (3 * wern + 1) / 4
    elif(type(rca) is RepeaterChainSampler):
        if(to_plot == 'pmf'):
            samples = rca.T_samples
            return _make_hists_from_samples(samples, trunc)
        elif(to_plot == 'cdf'):
            samples = rca.T_samples
            hists = _make_hists_from_samples(samples, trunc)
            ecdfs = np.zeros(shape=np.shape(hists))
            for i, hist in enumerate(hists):
                ecdfs[i] = prob_tools.pmf_to_cdf(hist)
            return ecdfs
        elif(to_plot == 'wern'):
            T_samples = rca.T_samples
            W_samples = rca.W_samples
            return _make_wern_functions_from_samples(T_samples,W_samples,trunc)
        elif(to_plot == 'fid'):
            T_samples = rca.T_samples
            W_samples = rca.W_samples
            werns = _make_wern_functions_from_samples(T_samples,W_samples,trunc)
            return (3 * werns + 1) / 4

def _make_hists_from_samples(samples, trunc):
    """Makes a distribution from given samples by putting them in a histogram.

    Parameters
    ----------
    samples : 2D array of samples[level, sample]
        Given samples
    trunc : string
        Max value of histogram.
    """
    levels = len(samples)
    hists = np.zeros(shape=(levels,trunc+1))
    for level, samples in enumerate(samples):
        hist, _ = np.histogram(samples, bins=range(trunc+3))
        hist = hist[:-1] # deals with inclusive right edge of last bin
        # normalize
        sample_size = len(samples)
        hists[level] = hist / sample_size
    return hists

def _make_wern_functions_from_samples(T_samples, W_samples, trunc):
    """Uses given samples to make the numerical W_n(t) functions.

    Parameters
    ----------
    T_samples : 2D array of samples[level, sample]
        Time samples.
    W_samples : 2D array of samples[level, sample]
        Werner parameter samples.
    trunc : string
        Max value of histogram.
    """
    levels  = len(T_samples)
    wern_fs = np.zeros(shape=(levels,trunc+1))
    for level in range(levels):
        sample_size = len(T_samples[level])
        counters = np.zeros(trunc+1)
        for samp in range(sample_size):
            wern = W_samples[level, samp]
            time = int(T_samples[level, samp])
            if(time <= trunc):
                wern_fs[level, time] += wern
                counters[time] += 1
        # counters[time] == 0 means no samples at that time, set
        # werner parameters for those times to NaN.
        wern_fs[level][counters == 0] = np.nan
        #counters[counters == 0] = 1
        wern_fs[level] /= counters
    return wern_fs

def _ecdf_conf_bands(ecdf, n_samps, alpha=0.01):
    """ Constructs a Dvoretzky-Kiefer-Wolfowitz confidence band for
    the given eCDF, uses DKW inequality.

    Parameters
    ----------
    ecdf : array
        Empirical cdf to construct bands around.
    n_samps : int
        Number of samples used to construct ecdf.
    alpha : float
        Confidence level (probability of actual CDF
        lying outside of bands <= alpha)

    Returns
    -------
    Tuple (upper : array, lower : array)
        Confidence bands for given empirical CDF.
    """
    epsilon = np.sqrt(np.log(2 / alpha) / (2 * n_samps))
    lower = np.clip(ecdf - epsilon, 0, 1)
    upper = np.clip(ecdf + epsilon, 0, 1)
    return lower, upper

def _plot_title(rca):
    """Plots distributions.

    Parameters
    ----------
    rca : RepeaterChainAnalyzer
        RepeaterChainAnalyzer object of which stuff is being plotted.

    Returns
    -------
    string
        A string containing some information to set as a title for the plot.
    """
    algorithm = ''
    if(type(rca) is RepeaterChainCalculator):
        algorithm = 'Deterministic algorithm'
    elif(type(rca) is RepeaterChainSampler):
        algorithm = 'Monte Carlo algorithm'
    return "{} - parameters: \n{}".format(algorithm, rca.params)

def _ylab(to_plot):
    """Returns the y-label for the plot given the type of plot.

    Parameters
    ----------
    to_plot : string
        Type of thing to plot. Can be 'pmf', 'cdf', 'fid', or 'wern'.

    Returns
    -------
    string
        The y-label for the plot.
    """
    labels_dict = {
        'pmf'  : "$\\Pr(T_n = t)$",
        'cdf'  : "$\\Pr(T_n \\leq t)$",
        'fid'  : "$F_n(t)$",
        'wern' : "$W_n(t)$"
    }
    return labels_dict[to_plot]
