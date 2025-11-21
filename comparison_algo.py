import numpy as np
import time

def mcmc_sample_density(
    density_fn,
    dim,
    N_samples,
    x0=None,
    proposal_std=1.0,
    burn_in=1000,
    thinning=1,
    random_state=None,
):
    """
    Generic Metropolis–Hastings MCMC sampler that only needs a *density* function.

    DIFFERENCE FROM PREVIOUS VERSION:
    - We now guarantee that the total number of *accepted* points returned is N_samples.
    - `burn_in` and `thinning` are interpreted in terms of ACCEPTED states:
        * First `burn_in` accepted states are discarded.
        * After that, we store every `thinning`-th accepted state
          until we have exactly N_samples stored.

    Parameters
    ----------
    density_fn : callable
        density_fn(x) -> float, proportional to target density.
        x is a 1D numpy array of shape (dim,). The density does NOT need to be
        normalized, but it must be non-negative and finite wherever called.
    dim : int
        Dimension of the state space.
    N_samples : int
        Number of accepted samples to return (after burn-in and thinning).
    x0 : array-like or None, optional
        Initial point for the chain. If None, starts at zeros.
    proposal_std : float, optional
        Standard deviation of isotropic Gaussian random-walk proposal:
            x' = x + N(0, proposal_std^2 I).
    burn_in : int, optional
        Number of *accepted* states to discard at the beginning of the chain.
    thinning : int, optional
        Keep one sample every `thinning` accepted states after burn-in.
    random_state : int or np.random.Generator or None, optional
        RNG seed or Generator.

    Returns
    -------
    samples : ndarray, shape (N_samples, dim)
        MCMC samples (accepted states) approximately from the target density.
    elapsed : float
        Wall-clock time in seconds.
    """

    start = time.perf_counter()

    # RNG
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # initial point
    if x0 is None:
        x_current = np.zeros(dim, dtype=float)
    else:
        x_current = np.asarray(x0, dtype=float).reshape(dim)

    samples = np.zeros((N_samples, dim), dtype=float)

    # current log-density (compute from density function)
    p_current = float(density_fn(x_current))
    if p_current <= 0.0 or not np.isfinite(p_current):
        raise ValueError("Initial point x0 has non-positive or invalid density.")
    logp_current = np.log(p_current)

    accepted_count = 0   # total accepted so far (including burn-in)
    stored_count = 0     # number of samples stored in `samples`

    while stored_count < N_samples:
        # propose new point
        noise = rng.normal(loc=0.0, scale=proposal_std, size=dim)
        x_prop = x_current + noise

        # compute log-density at proposal
        p_prop = float(density_fn(x_prop))
        if p_prop <= 0.0 or not np.isfinite(p_prop):
            # reject invalid or zero density automatically
            accept = False
        else:
            logp_prop = np.log(p_prop)

            # Metropolis–Hastings acceptance probability:
            # alpha = min(1, p_prop / p_current)
            # work in log-space for stability: log_alpha = logp_prop - logp_current
            log_alpha = logp_prop - logp_current

            if log_alpha >= 0.0:
                accept = True
            else:
                u = rng.uniform(0.0, 1.0)
                accept = (np.log(u) < log_alpha)

        if accept:
            x_current = x_prop
            logp_current = logp_prop
            accepted_count += 1

            # after burn-in, store every `thinning`-th accepted state
            if accepted_count > burn_in and ((accepted_count - burn_in) % thinning == 0):
                samples[stored_count, :] = x_current
                stored_count += 1

    elapsed = time.perf_counter() - start
    return samples, elapsed
