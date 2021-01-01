import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from protein_utils import *
from core.concentration import *
from risk_histogram import get_example_loss_and_size_tables
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb

def plot_protein_sets(pred, s, target):
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12,4))
    img = mpimg.imread('T0995_3WUY.png')
    axs[0].imshow(img)
    axs[0].set_title(f'3D rendering of {target}')
    axs[0].axis('off')
    axs[1].imshow(pred, cmap='Blues', vmin=0.0, vmax=20.0)
    axs[1].axis('off')
    axs[1].set_title(r'prediction ($\AA$)')
    axs[2].imshow(pred-s[0], cmap='Blues', vmin=0.0, vmax=20.0)
    axs[2].axis('off')
    axs[2].set_title(r'length of lower interval ($\AA$)')
    axp = axs[3].imshow(s[1]-pred, cmap='Blues', vmin=0.0, vmax=20.0)
    axs[3].axis('off')
    axs[3].set_title(r'length of upper interval ($\AA$)')

    #fig.subplots_adjust(right=0.80)
    cax = fig.add_axes([0.95, 0.16, 0.01, 0.66])

    cb = plt.colorbar(axp, cax=cax)

    fig.suptitle(f"Uncertainty sets for protein {target}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    plt.savefig('./outputs/protein_grid_fig.pdf')

if __name__ == "__main__":
    # Experimental parameters
    data_dir = '/scratch/aa/casp13-data/'
    gamma = 0.1 
    delta = 0.1
    num_lam = 1500 
    num_calib = 35 
    num_grid_hbb = 200
    epsilon = 1e-10 
    maxiters = int(1e5)
    num_trials = 1000 
    bound_str = 'CLT'
    bound_fn = get_bound_fn_from_string(bound_str)
    ub = 0.2
    ub_sigma = np.sqrt(2)
    lambdas_example_table = np.linspace(0,1,1000)
    deltas_precomputed = [0.001, 0.01, 0.05, 0.1]
    
    # Get the correct threshold
    ls_probs, ls_preds, ls_gt, ls_targets, ls_dists = get_preds_gt(data_dir)
    tlambda = get_tlambda(num_lam,deltas_precomputed,num_calib,num_grid_hbb,ub,ub_sigma,epsilon,maxiters,bound_str,bound_fn)
    example_loss_table, _ = get_example_loss_and_size_tables(ls_probs, ls_preds, ls_gt, ls_targets, ls_dists, lambdas_example_table, num_calib)
    example_loss_table = example_loss_table / 20. # it's bounded above by 20.
    calib_losses = example_loss_table[1:num_calib+1] # exclude the first example
    lhat = get_lhat_from_table(calib_losses[:,::-1], lambdas_example_table[::-1], gamma, delta, tlambda, bound_str)
    print(f"lhat:{lhat}")

    # predict
    ls_sets = ls_sets_from_probs(ls_probs, ls_preds, ls_dists, lhat)
    plot_protein_sets(ls_preds[0],ls_sets[0], ls_targets[0])
