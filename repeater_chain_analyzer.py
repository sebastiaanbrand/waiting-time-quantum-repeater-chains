import numpy as np
import pathlib
import json
import time
from scipy.stats import geom as scipy_geom
import probability_tools as prob_tools
import werner_tools as wern_tools

class RepeaterChainAnalyzer:
    """Base class for the RCCalculator and the RCSampler.

    This class should not be instantiated (but I'll leave that to the user
    rather than prevent doing this with ABC)
    """

    def __init__(self, outputfolder=None):
        """
        Parameters
        ----------
        outputfolder : string
            Path to the main output folder. If set to None results are not
            logged.
        """
        self.runtime = np.zeros(shape=(self.n+1))
        self.__init_logging(outputfolder)

    def __init_logging(self, outputfolder):
        """Initializes some stuff for logging if a folder is set.

        Parameterss
        ----------
        outputfolder : string
            Path to the main output folder. If set to None results are not
            logged.
        """
        if(outputfolder):
            self.outputfolder = outputfolder
            pathlib.Path(outputfolder).mkdir(parents=True, exist_ok=True)
            self.log_results = True
            self.log_params()
        else:
            self.log_results = False

    def log_params(self):
        """Logs the parameters of the repeater chain as JSON. """
        if(hasattr(self, 'params')):
            with open(self.outputfolder + "parameters.json", "w") as f:
                json.dump(self.params, f)

    def load_from_folder(self, folder):
        """Loads parameters from folder.

        Parameters
        ----------
        folder : string
            Folder to load from.
        """
        with open(folder + "parameters.json", "r") as f:
            self.params = json.load(f)

    def _print_time(self, n):
        """ Prints the amount of time it took to compute for the given level n.

        Requires self.runtime[n] to be set first.

        Parameters
        ----------
        n : int
            Level to print the time for.
        """
        time = self.runtime[n]
        print("Level {} - {} sec".format(n,round(time,2)))



class RepeaterChainCalculator(RepeaterChainAnalyzer):
    """Deterministic algorithm for calculating waiting time and fidelities. """

    def __init__(self, n, trunc, pgen, pswap, w0=None, T_coh=None, **kwds):
        """
        Parameters
        ----------
        n : int
            Number of BDCZ protocol levels, i.e. a repeater chain with 2^n
            segments.
        trunc : int
            Maximum number of timesteps t for which the probability Pr(T_n = t)
            is calculated.
        pgen : float
            Success probability of entanglement generatation between
            neighboring nodes.
        pswap : float
            Success probability of entanglement swap.
        w0 : float
            Werner parameter of the states generated between neighboring nodes.
            If set to None we only calculate waiting time and no fidelity.
        T_coh : float
            Memory coherence time. If set to 0 there is no memory decoherence.
        outputfolder : string
            Path to the main output folder. If set to None results are not
            logged.
        """
        self.params = {
            'pgen' : pgen,
            'pswap' : pswap,
            'w0' : w0,
            'T_coh' : T_coh
        }
        self.n     = n
        self.trunc = trunc
        self.pmf       = np.zeros(shape=(n+1, trunc+1))
        self.pmf_total = np.zeros(shape=(n+1, trunc+1))
        self.wern      = np.zeros(shape=(n+1, trunc+1))
        super().__init__(**kwds)

    def log_data(self):
        """Logs both the pmfs data, werner parameters, and runtimes, as csv. """
        datafolder = self.outputfolder + "data/"
        pathlib.Path(datafolder).mkdir(parents=True, exist_ok=True)
        np.savetxt(datafolder + "pmfs.csv", self.pmf, delimiter=',')
        np.savetxt(datafolder + "pmfs_total.csv",self.pmf_total, delimiter=',')
        np.savetxt(datafolder + "werns.csv", self.wern, delimiter=',')
        np.savetxt(datafolder + "runtime.csv", self.runtime, delimiter=',')

    def load_from_folder(self, folder):
        """Loads data from folder.

        Parameters
        ----------
        folder : string
            Folder to load from.
        """
        super().load_from_folder(folder)
        datafolder = folder + "data/"
        self.pmf       = np.loadtxt(datafolder + "pmfs.csv", delimiter=',')
        self.pmf_total = np.loadtxt(datafolder + "pmfs_total.csv",delimiter=',')
        self.wern      = np.loadtxt(datafolder + "werns.csv", delimiter=',')
        self.runtime   = np.loadtxt(datafolder + "runtime.csv", delimiter=',')
        self.n = len(self.pmf) - 1
        self.trunc = len(self.pmf[0]) - 1

    def calculate(self, verbose=True):
        """Calculates the waiting time and fidelities for all levels up to n.

        Waiting time pmfs are calculated and logged after every level.
        Fidelities are only calculated if 'w0' was set in the constructor.

        Parameters
        ----------
        verbose : boolean
            If set to True prints stuff to console.
        """
        if(verbose):
            print("Calculating distribution up to n={}".format(self.n))
        level = 0
        t_start = time.time()
        self.__set_ground_distribution()
        self.runtime[level] += time.time() - t_start
        if(verbose):
            self._print_time(level)
        for level in range(1, self.n+1):
            pmfs_swaps = self.__calculate_waiting_time(level)
            if(self.params['w0'] is not None):
                self.__calculate_werner(level, pmfs_swaps)
            self.__calculate_total_links(level)
            self.runtime[level] += time.time() - t_start
            if(verbose):
                self._print_time(level)
            if(self.log_results):
                self.log_data()

    def __set_ground_distribution(self):
        """Sets the waiting time pmf of T_0 to geom(pgen), and the sets the
        constant Werner parameter on the ground level if set.
        """
        for t in range(self.trunc+1):
            self.pmf[0,t] = scipy_geom.pmf(t, self.params['pgen'])
        self.pmf_total[0] = self.pmf[0]
        if(self.params['w0'] is not None):
            self.wern[0].fill(self.params['w0'])

    def __calculate_waiting_time(self, level):
        """Calculates the waiting time pmf[level] using pmf[level-1].

        Stores the calculated pmf in self.pmf[level]. Assumes self.pmf[level-1]
        is already calculated.

        Parameters
        ----------
        level : int
            Level to calculate the waiting time pmf for.
        """
        # Initialize some stuff
        pmf_in  = self.pmf[level-1]
        pmf_out = np.zeros(self.trunc+1)

        # Waiting for two parallel links can be computed via the square of the
        # cummulative distribution function.
        cdf_in  = prob_tools.pmf_to_cdf(pmf_in)
        cdf_max = prob_tools.max_distributions(cdf_in, 2)
        pmf_max = prob_tools.cdf_to_pmf(cdf_max)

        # The effect of the swap can be computed via a geometric sum.
        low_mem = True
        if(self.params['w0'] is not None):
            low_mem = False
        pmf_out, pmfs_swaps = prob_tools.random_geom_sum(pmf_max,
                                                         self.params['pswap'],
                                                         low_mem)

        # Update pmf
        self.pmf[level] = pmf_out

        # This intermediate computation result is needed for calculating
        # the Werner parameters, and we don't want to recompute this so we'll
        # pass it back
        return pmfs_swaps

    def __calculate_total_links(self, level):
        """Calculates the probability distribution of the total number of links.

        Stores the calculated pmf in self.pmf_total[level].
        Assumes self.pmf_total[level-1] is already calculated.

        The random variable of which this function computes the (truncated)
        distribution stochastically dominates the random variable of waiting
        time, and because we know the mean of _this_ random variable, we can
        use it to obtain numerical bounds on the mean of the waiting time.

        Parameters
        ----------
        level : int
            Level to calculate the pmf of the total number of links pmf for.
        """
        # Initialize some stuff
        pmf_in  = self.pmf_total[level-1]
        pmf_out = np.zeros(self.trunc+1)

        # Where in __calculate_waiting_time() we take the maximum of two random
        # variables (because these links are assumed to be generated in
        # parallel), here we add them to compute the total number of links.
        pmf_sum = prob_tools.sum_distributions(pmf_in, pmf_in)[:self.trunc+1]

        # The effect of the swap can be computed via a geometric sum.
        pmf_out, _ = prob_tools.random_geom_sum(pmf_sum, 
                                                self.params['pswap'],
                                                low_mem=True)

        # Update pmf
        self.pmf_total[level] = pmf_out

    def __calculate_werner(self, level, pmfs_swaps):
        """Calculates the Werner parameters wern[level] using wern[level-1].

        Stores the calculated Werner parameters in self.wern[level]. Assumes
        self.wern[level-1] and relevant probabilities are already calculated.

        Parameters
        ----------
        level : int
            Level to calculate the waiting time pmf for.
        pmfs_swaps : 2D NumPy array
            pmfs_swaps[s,t] = Pr(T_n = t | S = s), where T_n is the waiting time
            at the current level, and we condition on the number of swaps S.
        """
        W_in = self.wern[level-1]
        self.wern[level] = wern_tools.compute_werner_next_level(
            W_in=W_in,
            pmfs_swaps=pmfs_swaps,
            pmf_single=self.pmf[level-1],
            pswap=self.params['pswap'],
            T_coh=self.params['T_coh'],
            pmf_out=self.pmf[level]
        )

    def mean_bounds(self, level):
        """Computes lower and upper bounds on the mean.

        Uses pmf[level] and pmf_total[level] to compute an upper and
        lower bound on the mean. Requires having run calculate() first.

        Parameters
        ----------
        level : int
            Level to calculate mean bounds for.

        Returns
        -------
        Tuple (lower_bound : float, upper_bound : float)
        """
        lower_bound = self.mean_lower_bound(level)
        upper_bound = self.mean_upper_bound(level)
        return lower_bound, upper_bound

    def mean_lower_bound(self, level):
        """Computes a lower bound on the mean.

        Uses pmf[level] to compute a lower bound on the mean.
        Requires having run calculate() first.

        Parameters
        ----------
        level : int
            Level to calculate mean lower bound for.

        Returns
        -------
        float
            Lower bound on the mean of T_{level}.
            If there is no probability mass in the given pmf returns -1.
        """
        return prob_tools.numerical_mean(self.pmf[level])

    def mean_upper_bound(self, level, trunc=None):
        """Computes an upper bound on the mean.

        Uses pmf[level] and pmf_total[level] to compute an upper bound
        on the mean. Requires having run calculate() first.

        Parameters
        ----------
        level : int
            Level to calculate mean upper bound for.

        Returns
        -------
        float
            Upper bound on the mean of T_{level}.
        """
        if(trunc is None):
            trunc = self.trunc
        pmf_time  = self.pmf[level]
        pmf_total = self.pmf_total[level]
        pgen      = self.params['pgen']
        pswap     = self.params['pswap']

        latency_mass = np.sum(pmf_time[:trunc])
        attempt_mass = np.sum(pmf_total[:trunc])

        true_attempt_mean = (2/pswap)**level * (1/pgen)
        num_attempt_mean  = prob_tools.numerical_mean(pmf_total[:trunc])
        num_latency_mean  = prob_tools.numerical_mean(pmf_time[:trunc])
        if(attempt_mass < 1):
            nom = (true_attempt_mean - attempt_mass*num_attempt_mean)
            denom = (1-attempt_mass)
            tail_attempt_mean = nom / denom
            upper_bound = latency_mass*num_latency_mean + \
                          (1-latency_mass)*tail_attempt_mean
        else:
            upper_bound = num_latency_mean
        return upper_bound


class RepeaterChainSampler(RepeaterChainAnalyzer):
    """Monte Carlo approach to calculating waiting time and fidelities. """

    def __init__(self, n, pgen, pswap, comm_time=0,
                 w0=0, T_coh=0, n_dist=0, **kwds):
        """
        Parameters
        ----------
        n : int
            Number of BDCZ protocol levels, i.e. a repeater chain with 2^n
            segments.
        pgen : float
            Success probability of entanglement generatation between
            neighboring nodes.
        pswap : float
            Success probability of entanglement swap.
        comm_time : int or Python list of length (n+1)
            Communication time to do swaps. If set to an int the swaps on all
            levels take this amount of time. If set to a list, swaps on level k
            will take comm_time[k-1] time. Distillation of links on level k
            takes comm_time[k] time.
        w0 : float
            Werner parameter of the states generated between neighboring nodes.
        T_coh : float
            Memory coherence time.
            If set to None there is no memory decoherence.
        n_dist : int
            Number of distillation rounds per level. If set to 0 we are left
            with the BDCZ protocol without distillation.
        outputfolder : string
            Path to the main output folder. If set to None results are not
            logged.
        """
        if(type(comm_time) is int):
            comm_time = [comm_time] * (n+1)
        self.params = {
            'pgen' : pgen,
            'pswap' : pswap,
            'comm_time' : comm_time,
            'w0' : w0,
            'T_coh' : T_coh,
            'n_dist' : n_dist
        }
        self.n = n
        super().__init__(**kwds)

    def log_data(self):
        """Logs waiting time and werner samples, and runtimes, as csv. """
        datafolder = self.outputfolder + "data/"
        pathlib.Path(datafolder).mkdir(parents=True, exist_ok=True)
        np.savetxt(datafolder + "T_samples.csv", self.T_samples, delimiter=',')
        np.savetxt(datafolder + "W_samples.csv", self.W_samples, delimiter=',')

    def load_from_folder(self, folder):
        """Loads data from folder.

        Parameters
        ----------
        folder : string
            Folder to load from.
        """
        super().load_from_folder(folder)
        datafolder = folder + "data/"
        self.T_samples = np.loadtxt(datafolder + "T_samples.csv", delimiter=',')
        self.T_samples = np.array(self.T_samples, dtype=np.int)
        self.W_samples = np.loadtxt(datafolder + "W_samples.csv", delimiter=',')
        self.n = len(self.T_samples) - 1

    def sample(self, sample_size, verbose=True):
        """
        Samples `sample_size` number of samples of (times, wern) for repeater
        chains with 2^0, 2^1, ...,2^n segments, where n is given in the
        constructor. Logs results after every level.

        Parameters
        ----------
        sample_size : int
            Number of samples to draw for every level.
        verbose : boolean
            If set to True prints stuff to console.
        """
        if(verbose):
            print("Sampling {} up to level {}".format(sample_size, self.n))
        self.T_samples = np.zeros(shape=(self.n+1, sample_size), dtype=np.int)
        self.W_samples = np.zeros(shape=(self.n+1, sample_size))
        for level in range(0, self.n+1):
            t_start  = time.time()
            time_samps, wern_samps = self.sample_level(level, sample_size)
            self.T_samples[level] = time_samps
            self.W_samples[level] = wern_samps
            self.runtime[level] = time.time() - t_start
            if(verbose):
                self._print_time(level)
            if(self.log_results):
                self.log_data()


    def sample_level(self, n, sample_size):
        """ Samples tuples (time, wern) from a repeater chain with 2^n segments.

        Parameters
        ----------
        n : int
            Number of nesting levels of the repeater chain (i.e. 2^ segments).
        sample_size : int
            Number of samples to draw.

        Returns
        -------
        Tuple of two arrays (time_samples, wern_samples)
            time_samples[t] and wern_samples[t] correspond to the same link.
        """
        time_samples = np.zeros(sample_size)
        wern_samples = np.zeros(sample_size)
        for k in range(sample_size):
            time, wern = self.__sample_swap(n)
            time_samples[k] = time
            wern_samples[k] = wern
        return time_samples, wern_samples

    def __sample_swap(self, n):
        """
        Samples a single tuples (time, wern) from a
        repeater chain with 2^n segments.

        Parameters
        ----------
        n : int
            Number of nesting levels of the repeater chain (i.e. 2^ segments).

        Returns
        -------
        Tuple of (time : int, wern : float)
            The number of timesteps to generate the link and
            the corresponding Werner parameter.
        """
        if(n == 0):
            time = np.random.geometric(self.params['pgen'])
            wern = self.params['w0']
            return time, wern
        else:
            tA, wA = self.__sample_dist(n, self.params['n_dist'])
            tB, wB = self.__sample_dist(n, self.params['n_dist'])
            comm_time = self.params['comm_time'][n-1]
            time = max(tA, tB) + comm_time

            T_coh = self.params['T_coh']
            wA, wB = wern_tools.decohere_earlier_link(tA, tB, wA, wB, T_coh)
            wA = wern_tools.wern_after_memory_decoherence(wA, comm_time, T_coh)
            wB = wern_tools.wern_after_memory_decoherence(wB, comm_time, T_coh)
            wern = wern_tools.wern_after_swap(wA, wB)

            swap_success = np.random.random() <= self.params['pswap']
            if(swap_success):
                return time, wern
            else:
                time_retry, wern_retry = self.__sample_swap(n)
                return time + time_retry, wern_retry

    def __sample_dist(self, n, n_dist):
        """
        Samples a single tuples (time, wern) from a
        repeater chain with 2^n segments, where we do n_dist rounds of
        distillation on the current level, and self.params['n_dist'] rounds
        of distillation on the other levels.

        Parameters
        ----------
        n : int
            Number of nesting levels of the repeater chain (i.e. 2^ segments).
        n_dist : int
            Number of distillation rounds to do.

        Returns
        -------
        Tuple of (time : int, wern : float)
            The number of timesteps to generate the link and
            the corresponding Werner parameter.
        """
        if(n_dist == 0):
            return self.__sample_swap(n-1)
        else:
            tA, wA = self.__sample_dist(n, n_dist-1)
            tB, wB = self.__sample_dist(n, n_dist-1)
            comm_time = self.params['comm_time'][n]
            time = max(tA, tB) + comm_time
            T_coh = self.params['T_coh']
            wA, wB = wern_tools.decohere_earlier_link(tA, tB, wA, wB, T_coh)
            wA = wern_tools.wern_after_memory_decoherence(wA, comm_time, T_coh)
            wB = wern_tools.wern_after_memory_decoherence(wB, comm_time, T_coh)
            wern, p_dist = wern_tools.wern_after_distillation(wA, wB)
            dist_success = np.random.random() <= p_dist
            if(dist_success):
                return time, wern
            else:
                time_retry, wern_retry = self.__sample_dist(n, n_dist)
                return time + time_retry, wern_retry

    def sample_mean(self, level):
        """Calculates the sample mean and standard error.

        Parameters
        ----------
        level : int
            Level of which to get the sample mean E[T_{level}] for.

        Returns
        -------
        Tuple of (mean : float, standard_error : float)
        """
        samples = self.T_samples[level]
        mean    = np.mean(samples)
        se      = np.std(samples) / np.sqrt(len(samples))
        return mean, se
