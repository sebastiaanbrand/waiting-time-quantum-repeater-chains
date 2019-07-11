import numpy as np
from scipy.stats import geom as scipy_geom

def wern_after_swap(w1, w2):
    """Calculates the Werner parameter of a link after a swap on two links.

    Parameters
    ----------
    w1 : float
        Werner parameter of one of the two links.
    w2 : float
        Werner parameter of the other link.

    Returns
    -------
    float
        Werner parameter of the state after a swap on two links with w1 and w2.
    """
    return w1 * w2

def wern_after_distillation(wA, wB):
    """Calculates the Werner parameter and success probabililty of distillation.

    Parameters
    ----------
    wA : float
        Werner parameter of one of the two links.
    wB : float
        Werner parameter of the other link.

    Returns
    -------
    Tuple (wern, prob)
        Tuple of the Werner paramter after successful distillation, and the
        probability that distillation is indeed successful.
    """
    p_dist = (1 + wA * wB) / 2
    w_dist = (1 + wA + wB + 5 * wA * wB) / (6 * p_dist) - 1/3
    return w_dist, p_dist

def wern_after_memory_decoherence(w0, delta_t, T_coh):
    """Calculates the Werner parameter after delta_t time in memory.

    Parameters
    ----------
    w0 : float
        Werner parameter of the state at time 0.
    delta_t : float
        Time that has passed since time 0.
    T_coh : float
        Memory coherence time. If set to 0, there is no decay.

    Returns
    -------
    Werner parameter after the state has spend `delta_t` time in memory with
    memory coherence time `T_coh`.
    """
    if(T_coh == 0):
        return w0
    else:
        return w0 * np.exp(-delta_t / T_coh)

def decohere_earlier_link(tA, tB, wA, wB, T_coh):
    """Applies decoherence to the earlier generated of the two links.

    Parameters
    ----------
    tA : float
        Waiting time of one of the links.
    wA : float
        Corresponding fidelity
    tB : float
        Waiting time of the other link.
    wB : float
        Corresponding fidelity
    t_both : float
        Time both links experience decoherence (e.g. communication time)
    T_coh : float
        Memory coherence time. If set to 0, there is no decay.

    Returns
    -------
    Tuple (float : tA, float : tB, float : wA, float : wB) after decoherence.
    """
    delta_t = abs(tA - tB)
    if(tA < tB):
        wA = wern_after_memory_decoherence(wA, delta_t, T_coh)
    elif(tB < tA):
        wB = wern_after_memory_decoherence(wB, delta_t, T_coh)
    return wA, wB

def decoherence_and_swap(tA, tB, wA, wB, T_coh):
    """Calculates the Werner parameter after both decoherence and a swap.

    Parameters
    ----------
    tA : float
        Waiting time of one of the links.
    wA : float
        Corresponding fidelity
    tB : float
        Waiting time of the other link.
    wB : float
        Corresponding fidelity
    T_coh : float
        Memory coherence time. If set to 0, there is no decay.

    Returns
    -------
    Werner parameter after the state has spend `delta_t` time in memory with
    memory coherence time `T_coh`.
    """
    if(T_coh == 0):
        return wA * wB
    else:
        return wA * wB * np.exp( - abs(tA - tB) / T_coh )

def compute_werner_next_level(W_in, pmfs_swaps, pmf_single, pswap, T_coh,
                              pmf_out):
    """
    Computes the Werner parameters W[t] for the next level n + 1.

    Parameters
    ----------
    W_in : array of floats
        W_in[t] is the average Werner parameter of links delivered at time t
        at level n.
    pmfs_swaps : 2D-array of floats
        pmfs_swaps[s,t] = Pr(T_{n+1}=t | S=s)
    pmf_single : array of floats
        pmf_single[t] = Pr(T_n = t)
    pswap : float
        probability that a swap succeeds
    T_coh : float
        memory coherence time
    pmf_out : array of floats
        pmf_out[t] = Pr(T_{n+1} = t), only used to assert the correctness
        of the computed sums of probabilities

    Returns
    -------
    Array of floats
        W_out[t] is the average Werner parameter of links delivered at time t
        at level n+1.
    """
    trunc = len(W_in)

    W_out = np.zeros(trunc)
    probs = np.zeros(trunc)

    # Running all these loops from [0, trunc) produces correct results,
    # but we can optimize this a bit by not looping over combinations of values
    # where we know the probability is going to be 0 or t_deliver > trunc
    for s_fail in range(0, trunc):
        for t_fail in range(s_fail, trunc):
            for tA in range(1, trunc - t_fail):
                for tB in range(1, trunc - t_fail):

                    delta_t   = abs(tA - tB)
                    t_deliver = t_fail + max(tA, tB)
                    wA        = W_in[tA]
                    wB        = W_in[tB]
                    w_out     = decoherence_and_swap(tA, tB, wA, wB, T_coh)

                    # Pr(T_{n+1}=t_fail | S = s_fail)
                    if(s_fail == 0 and t_fail == 0):
                        prob = 1
                    elif(s_fail == 0 and t_fail > 0):
                        prob = 0
                    elif(s_fail >= 1):
                        prob  = pmfs_swaps[s_fail, t_fail]
                    prob *= scipy_geom.pmf(s_fail + 1, pswap) # Pr(S = s_succ)
                    prob *= pmf_single[tA] # Pr(T_n = t_a)
                    prob *= pmf_single[tB] # Pr(T_n = t_b)

                    # with the range of the loops this should always be the case
                    # check anyway
                    if(t_deliver < trunc):
                        W_out[t_deliver] += prob * w_out
                        probs[t_deliver] += prob

    assert(np.all(np.isclose(probs, pmf_out)))

    # normalize
    for t in range(trunc):
        if(probs[t] != 0):
            W_out[t] /= probs[t]
        else:
            W_out[t] = 0
    return W_out
