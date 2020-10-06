import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

def invert_bentkus(delta,R,n,B):
    binoinv = stats.binom.ppf(delta/np.e,n,R/B)
    return (n*R-B*binoinv)/n

def invert_hoeffding(delta,R,n,B):
    return np.sqrt(-np.log(delta)/2/n)*B**2

def map_bounds_R(bnds,Rs,delta,n,B):
    out = []
    for bnd in bnds:
        bnd_out = np.zeros((Rs.shape[0],)) 
        for i in range(Rs.shape[0]):
            bnd_out[i] = (Rs[i]-bnd(delta,Rs[i],n,B))/Rs[i] # The percentage of R you need to have.  
        out = out + [bnd_out]
    return out

def gridplot_bounds(bnds,Rs,deltas,ns,B,xlims,ylims,labels):
    fig, axs = plt.subplots(nrows=ns.shape[0],ncols=deltas.shape[0],sharex='col',sharey='row')
    for i in range(ns.shape[0]):
        for j in range(deltas.shape[0]):
            curves = map_bounds_R(bnds,Rs,deltas[j],ns[i],B)
            for k in range(len(curves)):
                axs[i,j].plot(Rs,curves[k],label=labels[k])
            axs[i,j].set_ylim(bottom=ylims[0],top=ylims[1])
            axs[i,j].set_xlim(left=xlims[0],right=xlims[1])
            axs[i,j].set_xscale('log')
            axs[i,j].hlines(0.8,min(Rs),max(Rs),linestyles='dashed',color='#11111111')
            axs[i,j].set_xticks([1e-3,1e-2,1e-1])
            axs[i,j].set_yticks([0, 0.5, 1])
            plt.setp(axs[i,j].get_xticklabels(),fontsize=10)
            plt.setp(axs[i,j].get_yticklabels(),fontsize=10)
            if i == 0:
                axs[i,j].set_title(r'$\delta$='+str(deltas[j]))
            if j == deltas.shape[0]-1:
                axs[i,j].yaxis.set_label_position("right")
                axs[i,j].set_ylabel('n='+str(ns[i]),labelpad=20,rotation=270)
            if i == 0 and j == deltas.shape[0]-1: 
                lgd = axs[i,j].legend()

    sns.despine(top=True,right=True)
    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("R")
    plt.ylabel("Efficiency, (R-t)/R")
    
    plt.savefig('../outputs/concentration_efficiency.pdf')

if __name__ == "__main__":
    sns.set(palette='pastel',font='serif')
    sns.set_style('white')
    deltas = np.array([0.1,0.01,0.001])
    ns = np.array([1000,10000,100000])
    B = 1
    Rs = np.linspace(0.0001,0.1,2000)
    ylims = [0,1]
    xlims = [min(Rs),max(Rs)]

    bnds = (invert_bentkus,invert_hoeffding)
    labels = ["Bentkus","Hoeffding"]

    gridplot_bounds(bnds,Rs,deltas,ns,B,xlims,ylims,labels)
