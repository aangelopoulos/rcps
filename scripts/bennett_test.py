import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from core.bounds import empirical_bennett_mu_plus
import numpy as np
from scipy.optimize import brentq

if __name__ == "__main__":
    n_cal = int(1e5)
    n_reps = int(1e6)
    maxiters = int(1e5)
    num_grid_bennett = 1000
    mus = [0.05, 0.1, 0.2]
    deltas = [0.2, 0.1, 0.05, 0.01, 0.001]

    delta = .1
    muhat = .1
    ucb = empirical_bennett_mu_plus(muhat, np.sqrt(2*muhat*(1-muhat)), n_cal, delta, num_grid_bennett, maxiters) # 1 and 100 are dummy arguments.
    x = np.random.binomial(n_cal,ucb,size=(n_reps,))/n_cal
    print( (x <= muhat).mean() * np.e / delta ) # Should be near 1

    for mu in mus:
        for delta in deltas:
            print(f"mu: {mu}, delta: {delta}")
            def _to_invert(muhat):
                sigmahat = np.sqrt(2*muhat*(1-muhat))
                return empirical_bennett_mu_plus(muhat, sigmahat, n_cal, delta, num_grid_bennett, maxiters) - mu
            thresh = brentq(_to_invert, 1e-10, mu, maxiter=maxiters) 
            x = np.random.binomial(n_cal,mu,size=(n_reps,))/n_cal
            print(f"empirical/theory: { (x <= thresh).mean() * np.e / delta }")
