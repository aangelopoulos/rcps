import numpy as np
from scipy.stats import binom
from scipy.optimize import brenth

def h1(y, mu):
    return y*np.log(y/mu) + (1-y)*np.log((1-y)/(1-mu))

def h2(y):
    return (1+y)*np.log(1+y) - y

### Log upper-tail inequalities of mean
def hoeffding_plus(mu, x, n):
    return -n * h1(np.maximum(mu,x),mu)

def bennett_plus(mu, sigma, x, n, num_grid):
    gamma = np.linspace(0, mu * (1-1/num_grid), num_grid)
    v = (sigma**2 + gamma**2)
    b = mu - gamma
    t = np.maximum(mu-x,0) # could replace with np.zeros_like(mu)
    res = v/b**2 * h2(b*t/v)    
    return -2 * np.max(res)

def bentkus_plus(mu, x, n):
    return np.log(binom.cdf(np.ceil(n*x),n,mu))+1

### Log upper-tail inequalities of emprical variance
def hoeffding_var(sigma2,x,n):
    m = np.floor(n/2)
    return -m * h1(np.minimum(sigma2,x),sigma2)

def bentkus_var(sigma2, x, n):
    m = np.floor(n/2)
    return np.log(binom.cdf(np.ceil(m*x),m,sigma2))+1

def maurer_pontil_var(sigma2, x, n):
    return -(n-1)/2/sigma2 * np.maximum(sigma2-x,0)**2

### Upper confidence bound of empirical variance via Hoeffding-Bentkus-Maurer-Pontil inequality
def HBMP_sigma_plus(sigmahat,n,alpha):
    sigma2hat = sigmahat**2
    def _tailprob(sigma2):
        hoeffding_sigma = hoeffding_var(sigma2,sigma2hat,n)
        bentkus_sigma=bentkus_var(sigma2,sigma2hat,n)
        maurer_pontil_sigma = maurer_pontil_var(sigma2, sigma2hat, n)
        return min(hoeffding_sigma, bentkus_sigma, maurer_pontil_sigma)-np.log(alpha)
    if _tailprob(0.25) > 0:
        return 0.5
    else:
        return np.sqrt(brenth(_tailprob,sigma2hat, 0.25))

### Upper confidence bound of mean via Hoeffding-Bentkus-Empirical Bennett inequalities
def HBB_mu_plus(muhat,sigmahat,n,alpha,num_grid):
    sigmahat_plus = HBMP_sigma_plus(sigmahat, n, alpha/2)
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu,muhat,n)
        bentkus_mu = bentkus_plus(mu,muhat,n)
        bennett_mu = bennett_plus(mu,sigmahat_plus,muhat,n,num_grid)
        return min(hoeffding_mu,bentkus_mu, bennett_mu) - np.log(alpha / 2)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brenth(_tailprob,muhat, 1-1e-10)

if __name__ == "__main__":
    print(HBB_mu_plus(0.9,0.01,100,0.05,100))
