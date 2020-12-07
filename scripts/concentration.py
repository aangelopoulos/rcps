import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import core.bounds as bounds
import seaborn as sns
import pdb

def map_bounds_R(bnds,Rs,delta,n,B,num_grid,sigmahat_factor,maxiters):
    Rs = Rs/B
    out = []
    for bnd in bnds:
        bnd_out = np.zeros((Rs.shape[0],)) 
        for i in range(Rs.shape[0]):
            R_plus_t = bnd(Rs[i], sigmahat_factor*np.sqrt(Rs[i]*(1-Rs[i])), n, delta, num_grid, maxiters)
            #if i == int(Rs.shape[0]/2):
            #    pdb.set_trace()
            bnd_out[i] = max((2*Rs[i] - R_plus_t)/Rs[i],0) # (R-t)/R : the percentage of R you need to have.  
        out = out + [bnd_out]
    return out

def gridplot_bounds(bnds,Rs,deltas,ns,B,num_grid,sigmahat_factor,xlims,ylims,labels,maxiters):
    fig, axs = plt.subplots(nrows=ns.shape[0],ncols=deltas.shape[0],sharex='col',sharey='row',figsize=(2*(6.5),2*ns.shape[0]))
    for i in range(ns.shape[0]):
        for j in range(deltas.shape[0]):
            curves = map_bounds_R(bnds,Rs,deltas[j],ns[i],B,num_grid,sigmahat_factor,maxiters)
            for k in range(len(curves)):
                line = axs[i,j].plot(Rs,curves[k],label=labels[k], alpha=1,linewidth=3)
            axs[i,j].set_ylim(bottom=ylims[0],top=ylims[1])
            axs[i,j].set_xlim(left=xlims[0],right=xlims[1])
            axs[i,j].set_xscale('log')
            axs[i,j].hlines(0.8,min(Rs),max(Rs),linestyles='dashed',color='#333333',label='80% efficient')
            axs[i,j].set_xticks([1e-3,1e-2,1e-1])
            axs[i,j].set_yticks([0, 0.5, 1])
            plt.setp(axs[i,j].get_xticklabels(),fontsize=12)
            plt.setp(axs[i,j].get_yticklabels(),fontsize=12)
            if i == 0:
                axs[i,j].set_title(r'$\delta$='+str(deltas[j]), fontdict={'fontsize':12})
            if j == deltas.shape[0]-1:
                axs[i,j].yaxis.set_label_position("right")
                axs[i,j].set_ylabel('n='+str(ns[i]),labelpad=20,rotation=270, fontsize=12)
            if i == 0 and j == deltas.shape[0]-1: 
                lgd = axs[i,j].legend(loc='upper left', bbox_to_anchor=(1.08,1), prop={'size':8})

    sns.despine(top=True,right=True)
    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("R", fontsize=12)
    plt.ylabel("Efficiency, (R-t)/R", fontsize=12)
    fig.tight_layout()
    
    plt.savefig('../' + (f'outputs/concentration_efficiency/{sigmahat_factor}_concentration_efficiency').replace('.','_') + '.pdf')

if __name__ == "__main__":
    sns.set(palette='Set1',font='serif')
    sns.set_style('white')
    deltas = np.array([0.1,0.01,0.001])
    ns = np.array([1000,10000,100000])
    B = 1
    Rs = np.logspace(-3,-0.5,50)
    ylims = [0,1]
    xlims = [min(Rs),max(Rs)]
    sigmahat_factors = [0.1, 0.4, 1] 
    num_grid = 100
    dash_len = 3 
    maxiters = int(1e5) 

    bnds = (bounds.hoeffding_naive_mu_plus, bounds.bentkus_mu_plus, bounds.empirical_bennett_mu_plus, bounds.HBB_mu_plus)
    labels = ["Hoeffding", "Bentkus", "Bennett", "HBB"]

    for sigmahat_factor in sigmahat_factors:
        gridplot_bounds(bnds,Rs,deltas,ns,B,num_grid,sigmahat_factor,xlims,ylims,labels,maxiters)
