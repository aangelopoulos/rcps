import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from core.bounds import WSR_mu_plus
from core.concentration import get_tlambda, get_lhat_from_table, get_lhat_from_table_binarysearch 
import numpy as np
from scipy.optimize import brentq
from tqdm import tqdm
import pdb

def get_coco_example_loss_and_size_tables(lambdas_example_table):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'../coco/.cache/{lam_low}_{lam_high}_{lam_len}_example_loss_table.npy'
    fname_sizes = f'../coco/.cache/{lam_low}_{lam_high}_{lam_len}_example_size_table.npy'
    loss_table = np.load(fname_loss)
    sizes_table = np.load(fname_sizes)
    return loss_table, sizes_table

if __name__ == "__main__":
    n_cal = int(4000)
    dataset_replicates = 5 
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
    
    # get losses
    example_loss_table, _ = get_coco_example_loss_and_size_tables(lambdas_table)
    example_loss_table = np.concatenate( (example_loss_table,)*dataset_replicates, axis=0 )
    example_loss_table = example_loss_table + np.random.uniform(size=example_loss_table.shape)/100
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
    print(f"dataset replicates: {dataset_replicates}")
    print((risks > gamma).mean())
    print(risks)
