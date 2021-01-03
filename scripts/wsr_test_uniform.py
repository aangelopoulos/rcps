import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from core.bounds import WSR_mu_plus
from core.concentration import get_tlambda, get_lhat_from_table, get_lhat_from_table_binarysearch 
import numpy as np
from scipy.optimize import brentq
from tqdm import tqdm
import pdb

if __name__ == "__main__":
    n_cal = int(4000)
    n_val = int(1000)
    n_lambda = 10000
    n_reps = int(1e2)
    epsilon = 1e-10
    maxiters = int(1e5)
    num_grid_bennett = 1000
    mus = [0.05, 0.1, 0.2]
    deltas = [0.001, 0.01, 0.05, 0.1]
    lambdas_table = np.linspace(0,1,n_lambda)

    delta = .1
    gamma = .1
    
    # set up losses to ramp from 0 to 1 linearly.
    example_loss_table = np.random.uniform(size=(n_cal+n_val,n_lambda))
    np.cumsum(example_loss_table,axis=1, out=example_loss_table)
    example_loss_table[:,1:] = example_loss_table[:,:-1]
    example_loss_table[:,0] = 0
    example_loss_table = example_loss_table / (example_loss_table.max(axis=1)[:,None]) 
    example_loss_table = example_loss_table[:,::-1]
    risks = np.zeros((n_reps,))
    # get the bound
    bound_str = 'WSR'
    bound_fn = WSR_mu_plus
    tlambda = get_tlambda(1500,deltas,n_cal,None,None,None,epsilon,maxiters,bound_str,bound_fn)
    for j in tqdm(range(n_reps)): 
        np.random.shuffle(example_loss_table)
        calib_loss_table, val_loss_table = (example_loss_table[:n_cal], example_loss_table[n_cal:])

        # get lhat (should be close to gamma)
        lhat = get_lhat_from_table_binarysearch(calib_loss_table, lambdas_table, gamma, delta, tlambda, bound_str)

        val_losses = val_loss_table[:,np.argmax(lambdas_table == lhat)]
        risks[j] = val_losses.mean()
    print((risks > gamma).mean())
    pdb.set_trace()
    print(risks)
    #sigmahat = np.sqrt(2*muhat*(1-muhat))
    #ucb = HBB_mu_plus(muhat, sigmahat, n_cal, delta, num_grid_bennett, maxiters) # 1 and 100 are dummy arguments.
    #x = np.random.binomial(n_cal,ucb,size=(n_reps,))/n_cal
    #print( (x <= muhat).mean() * np.e / delta ) # Should be near 1

    #for mu in mus:
    #    for delta in deltas:
    #        print(f"mu: {mu}, delta: {delta}")
    #        def _to_invert(muhat):
    #            sigmahat = np.sqrt(2*muhat*(1-muhat))
    #            return HBB_mu_plus(muhat, sigmahat, n_cal, delta, num_grid_bennett, maxiters) - mu
    #        thresh = brentq(_to_invert, 1e-10, mu, maxiter=maxiters) 
    #        x = np.random.binomial(n_cal,mu,size=(n_reps,))/n_cal
    #        print(f"empirical/theory: { (x <= thresh).mean() * np.e / delta }")
