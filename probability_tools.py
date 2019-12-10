import numpy as np
from scipy.stats import geom as scipy_geom
from scipy.signal import fftconvolve

def pmf_to_cdf(pmf):
    """Translates a PMF to the corresponding CDF.

    Parameters
    ----------
    pmf : array
        Probability mass function of some r.v. X such that pmf[x] = Pr(X=x).

    Returns
    -------
    NumPy array of the same length as `pmf`.
        Cumulative distribution function of the same r.v. X such that
        return[x] = Pr(X <= x).
    """
    return np.cumsum(pmf)

def cdf_to_pmf(cdf):
    """Translates a CDF to the corresponding PMF.

    Parameters
    ----------
    cdf : array
        Cumulative distribution function of some r.v. X such that
        cdf[x] = Pr(X <= x).

    Returns
    -------
    NumPy array of the same length as `cdf`.
        Probability mass function of the same r.v. X such that
        return[x] = Pr(X = x).
    """
    return np.diff(np.hstack((0,cdf)))

def numerical_mean(pmf):
    """Calculates the mean of a given (truncated) pmf, supposing this is all the
    probability mass. Acts as a lower bound on the mean of the untruncated
    distribution.

    Parameters
    ----------
    pmf : NumPy array
        (Possibly truncated) probability mass function such that pmf[x] = Pr(x)

    Returns
    -------
    float
        Mean of given pmf, after this pmf has been normalized to sum to 1.
        If there is no probability mass in the given pmf returns -1.
    """
    mass = np.sum(pmf)
    if(mass > 0):
        return np.sum(np.multiply(pmf, range(0,len(pmf))))/mass
    else:
        return -1

def max_distributions(cdf, m):
    """ Computes the CDF of `m` i.i.d. r.v.'s X distributed according to `cdf`.

    Parameters
    ----------
    cdf : NumPy array
        Cumulative distribution such that cdf[x] = Pr(X <= x).
    m : int
        Number of random variables X_1, ... X_m to take maximum of.

    Returns
    -------
    NumPy array of the same lenght as `cdf`.
        Cumulative distribution function of Z = max{X_1, X_2, ... X_m}.
    """
    return cdf ** m


def sum_distributions(pmf1, pmf2):
    """ Computes the PMF of Z = X + Y, for independent X and Y.

    Parameters
    ----------
    pmf1 : array
        Probability distribution of X such that pmf[x] = Pr(X = x).
    pmf2 : array
        Probability distribution of Y such that pmf[y] = Pr(Y = y).

    Returns
    -------
    NumPy array of length len(pmf1) + len(pmf2)
        Probability distribution of Z = X + Y such that return[z] = Pr(Z = z).
    """
    if(min(len(pmf1),len(pmf2)) >= 1000):
        return fftconvolve(pmf1, pmf2)
    else:
        return np.convolve(pmf1, pmf2)

def random_geom_sum(pmf, p, low_mem=False):
    """Calculates the distribution of Z = X_1 + X_2 + ... + X_N.

    Parameters
    ----------
    pmf : array
        Probability distribution of X such that pmf[x] = Pr(X = x).
    p : float
        Probability such that N ~ geom(p), i.e. Pr(N = n) = p(1-p)^{n-1}.
    low_mem : boolean
        If set to True this function doesn't store or output the intermediate
        results `pmf_given_N`, saving a lot of memory. Note that when next to
        calculating the waiting time, the Werner parameters are calculated as
        well, this value must be set to False, because the results of
        `get_pmfs_after_fixed_lengths(pmf)` are required for the Werner
        parameter calculation.
        NOTE: refactoring would also allow for a lower memory implementation for
        the Werner parameter calculation as well.

    Returns
    -------
    Tuple (pmf_out, pmfs_given_N)
        pmf_out[z]       = Pr(sum^N X = z) = Pr(Z = z).
        pmf_given_N[n,z] = Pr(sum^n X = z) = Pr(Z = z | N = n).
    """
    if(low_mem):
        pmf_final = get_pmf_after_prob_length_low_memory(pmf, p)
        pmfs_given_N = None
    else:
        pmfs_given_N = get_pmfs_after_fixed_lengths(pmf)
        pmf_final    = get_pmf_after_prob_length(pmfs_given_N, p)
    return pmf_final, pmfs_given_N

def get_pmfs_after_fixed_lengths(pmf):
    """
    Parameters
    ----------
    pmf : 1D numpy array
        Input pmf (possibly truncated) of some random variable X,
        such that pmf[x] = Pr(X = x).

    Returns
    -------
    2D (square) numpy array
        A square array `pmfs_fixed_lenghts` such that pmfs[k,x] =
        Pr(sum_{j=1}^k X_j = x), where X_j are i.i.d. random variables ~ X.
    """
    trunc  = len(pmf)
    res    = np.zeros(shape=(trunc,trunc))
    res[1] = pmf
    for k in range(2,trunc):
        res[k] = sum_distributions(res[k-1], pmf)[:trunc]
    return res

def get_pmf_after_prob_length(pmfs_fixed_lenghts, p):
    """
    Parameters
    ----------
    pmfs_fixed_lenghts : 2D (square) numpy array
        A square array `pmfs_fixed_lenghts` such that pmfs[k,x] =
        Pr(sum_{j=1}^k X_j = x), where X_j are i.i.d. random variables ~ X.
    p : float
        Probability such that the lenght of this sum ~ geom(p).

    Returns
    -------
    1D numpy array
        Probability mass function `pmf` such that
        pmf[x] = Pr(sum_{j=1}^N X_j = x), with N ~ geom(p)
    """
    trunc = len(pmfs_fixed_lenghts[0])
    pmf_final = np.zeros(trunc)
    for s in range(1, trunc):
        pmf_final = np.add(pmf_final, scipy_geom.pmf(s,p)*pmfs_fixed_lenghts[s])
    return pmf_final

def get_pmf_after_prob_length_low_memory(pmf, p):
    """
    This single function acts as a low memory version of the following two
    functions combined.
    - get_pmfs_after_fixed_lengths(pmf)
    - get_pmf_after_prob_length(pmfs_fixed_lenghts, p)
    The lower memory usage, O(trunc) instead of O(trunc^2), is achieved by not
    storing all the intermediate results from `get_pmfs_after_fixed_lengths()`.

    Parameters
    ----------
    pmf : 1D numpy array
        Input pmf (possibly truncated) of some random variable X,
        such that pmf[x] = Pr(X = x).
    p : float
        Probability such that the lenght of this sum ~ geom(p).

    Returns
    -------
    1D numpy array
        Probability mass function `pmf` such that
        pmf[x] = Pr(sum_{j=1}^N X_j = x), with N ~ geom(p)
    """
    trunc = len(pmf)
    pmf_temp  = np.copy(pmf)
    pmf_final = np.zeros(trunc)
    for s in range(1, trunc):
        if(scipy_geom.pmf(s,p) == 0):
            break
        pmf_final = np.add(pmf_final, scipy_geom.pmf(s,p)*pmf_temp)
        pmf_temp  = sum_distributions(pmf_temp, pmf)[:trunc]
    return pmf_final
