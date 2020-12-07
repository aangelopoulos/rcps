import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import core.bounds as bounds
import seaborn as sns
import pdb

def map_pinelis_utev(deltas,n,cvs,maxiters):
    out = []
    for cv in cvs:
        bnd_out = np.zeros((deltas.shape[0],)) 
        for i in range(deltas.shape[0]):
            # muhat, n, delta, cv
            R = 0.1 # The value of R doesn't matter!
            R_plus_t = bounds.pinelis_utev_mu_plus(R, n, deltas[i], cv, maxiters)
            bnd_out[i] = max((2*R - R_plus_t)/R,0) # (R-t)/R : the percentage of R you need to have.  
        out = out + [bnd_out]
    return out

def gridplot_bounds(deltas,ns,cvs,xlims,ylims,labels,maxiters):
    fig, axs = plt.subplots(nrows=1,ncols=ns.shape[0],sharex='col',sharey='row',figsize=(2*(6.5),2))
    for i in range(ns.shape[0]):
        curves = map_pinelis_utev(deltas,ns[i],cvs,maxiters)
        for k in range(len(curves)):
            line = axs[i].plot(deltas,curves[k],label=labels[k], alpha=1,linewidth=3)
        axs[i].set_ylim(bottom=ylims[0],top=ylims[1])
        axs[i].set_xlim(left=xlims[0],right=xlims[1])
        axs[i].set_xscale('log')
        axs[i].hlines(0.8,min(deltas),max(deltas),linestyles='dashed',color='#333333',label='80% efficient')
        axs[i].set_xticks([1e-3,1e-2,1e-1,1])
        axs[i].set_yticks([0, 0.5, 1])
        plt.setp(axs[i].get_xticklabels(),fontsize=12)
        plt.setp(axs[i].get_yticklabels(),fontsize=12)
        axs[i].set_title(r'$n=$'+str(ns[i]), fontdict={'fontsize':12})
        if i == ns.shape[0]-1: 
            lgd = axs[i].legend(loc='upper left', bbox_to_anchor=(1.08,1), prop={'size':8})

    sns.despine(top=True,right=True)
    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(r'$\delta$', fontsize=12)
    plt.ylabel("Efficiency\n(R-t)/R\n", fontsize=12)

    fig.tight_layout()
    
    plt.savefig('../outputs/concentration_efficiency/pinelis_utev.pdf')

if __name__ == "__main__":
    sns.set(palette='Set1',font='serif')
    sns.set_style('white')
    deltas = np.logspace(-3,0,50)
    ns = np.array([1000,10000,100000])
    cvs = np.array([0.1, 1, 10])
    ylims = [0,1]
    xlims = [min(deltas),max(deltas)]
    num_grid = 100
    dash_len = 3 
    maxiters = int(1e5) 

    labels = ["cv=.1", "cv=1", "cv=10"]

    gridplot_bounds(deltas,ns,cvs,xlims,ylims,labels,maxiters)
