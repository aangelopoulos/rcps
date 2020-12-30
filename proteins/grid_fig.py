import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from protein_utils import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb

def plot_protein_sets(pred, s, target):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9,4))
    axs[0].imshow(pred, cmap='Blues', vmin=0.0, vmax=20.0)
    axs[0].axis('off')
    axs[0].set_title(r'prediction ($\AA$)')
    axs[1].imshow(pred-s[0], cmap='Blues', vmin=0.0, vmax=20.0)
    axs[1].axis('off')
    axs[1].set_title(r'length of LCB ($\AA$)')
    axp = axs[2].imshow(s[1]-pred, cmap='Blues', vmin=0.0, vmax=20.0)
    axs[2].axis('off')
    axs[2].set_title(r'length of UCB ($\AA$)')

    #fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.20, 0.05, 0.58])

    cb = plt.colorbar(axp, cax=cax)

    fig.suptitle(f"{target} distogram", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    plt.savefig('./outputs/protein_grid_fig.pdf')

if __name__ == "__main__":
    lam = 0.07
    data_dir = '/scratch/aa/casp13-data/'
    ls_probs, ls_preds, ls_gt, ls_targets, ls_dists = get_preds_gt(data_dir)
    ls_sets = ls_sets_from_probs(ls_probs, ls_preds, ls_dists, lam)
    plot_protein_sets(ls_preds[0],ls_sets[0], ls_targets[0])
