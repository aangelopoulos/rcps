import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from core.bounds import empirical_bennett_mu_plus
import numpy as np
from scipy.optimize import brentq
import pdb 

if __name__ == "__main__":
    n_cal = int(1e5)
    n_reps = int(1e6)
    maxiters = int(1e5)
    num_grid_bennett = 1000
    mus = [0.05, 0.1, 0.2]
    deltas = [0.2, 0.1, 0.05, 0.01, 0.001]
    epsilon = 1e-10
    ub = 0.2 
    ub_sigma = np.sqrt(2)
    npts = 300 

    # Grid stuff
    rhats_grid = np.linspace(epsilon,ub,npts)
    sigmas_grid = np.linspace(epsilon,ub_sigma,max(int(npts/10),1))
    #rhats_grid = np.logspace(np.log10(epsilon),np.log10(ub),npts)
    #sigmas_grid = np.logspace(np.log10(epsilon),np.log10(np.sqrt(2)),max(int(npts/10),1))

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

                r = min(int(np.floor(muhat/ub * npts)),npts-1)
                s = min(int(np.ceil(sigmahat/ub_sigma * max(int(npts/10),1))),max(int(npts/10),1)-1)

                #r = min(int(np.floor((np.log10(muhat)-np.log10(epsilon))/(np.log10(ub)-np.log10(epsilon)) * npts)),npts-1)
                #s = min(int(np.ceil((np.log10(sigmahat)-np.log10(epsilon))/(np.log10(np.sqrt(2))-np.log10(sigmahat)) * max(int(npts/10),1))),max(int(npts/10),1)-1)

                #print(f"Rhat: {muhat} | Rhat_disc: {rhats_grid[r]} | sigmahat: {sigmahat} | sigmahat_disc: {sigmas_grid[s]}")

                return empirical_bennett_mu_plus(rhats_grid[r], sigmas_grid[s], n_cal, delta, num_grid_bennett, maxiters) - mu

            Rhat = np.random.binomial(n_cal,mu,size=(n_reps,))/n_cal

            thresh = brentq(_to_invert, epsilon, mu, maxiter=maxiters) 

            print(f"empirical/theory: { (Rhat <= thresh).mean() * np.e / delta }")
