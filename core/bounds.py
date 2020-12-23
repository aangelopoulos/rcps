import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
import pdb

def h1(y, mu):
    return y*np.log(y/mu) + (1-y)*np.log((1-y)/(1-mu))

def h2(y):
    return (1+y)*np.log(1+y) - y

### Log tail inequalities of mean
def hoeffding_naive(mu, x, n):
    return -n * (x-mu)**2 /2 

def hoeffding_plus(mu, x, n):
    return -n * h1(np.maximum(mu,x),mu)

def bennett_plus(mu, sigma, x, n, num_grid):
    gamma = np.linspace(0, mu * (1-1/num_grid), num_grid)
    v = (sigma**2 + gamma**2)
    b = mu - gamma
    t = np.maximum(mu-x,0) # could replace with np.zeros_like(mu)
    res = v/b**2 * h2(b*t/v)    
    return -n * np.max(res)

def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.floor(n*x),n,mu),1e-10))+1

def pinelis_utev(mu, x, n, cv):
    return -n/(cv**2 + 1) * (1 + x/mu * np.log( x/(np.e*mu) ) )

### Log upper-tail inequalities of emprical variance
def hoeffding_var(sigma2, x, n):
    m = np.floor(n/2)
    return -m * h1(np.minimum(sigma2,x),sigma2)

def bentkus_var(sigma2, x, n):
    m = np.floor(n/2)
    return np.log(max(binom.cdf(np.ceil(m*x),m,sigma2),1e-10))+1

def maurer_pontil_var(sigma2, x, n):
    return -(n-1)/2/sigma2 * np.maximum(sigma2-x,0)**2

### Upper confidence bound of empirical variance via Hoeffding-Bentkus-Maurer-Pontil inequality
def HBMP_sigma_plus(sigmahat, n, delta, maxiters): 
    sigma2hat = sigmahat**2 
    def _tailprob(sigma2): 
        hoeffding_sigma = hoeffding_var(sigma2, sigma2hat, n) 
        bentkus_sigma = bentkus_var(sigma2, sigma2hat, n)
        maurer_pontil_sigma = maurer_pontil_var(sigma2, sigma2hat, n)
        return min(hoeffding_sigma, bentkus_sigma, maurer_pontil_sigma)-np.log(delta)
    if _tailprob(0.25) > 0:
        return 0.5
    else:
        return np.sqrt(brentq(_tailprob, sigma2hat, 0.25,maxiter=maxiters))

### Upper confidence bound of mean via Hoeffding-Bentkus-Empirical Bennett inequalities
def HBB_mu_plus(muhat, sigmahat, n, delta, num_grid, maxiters):
    sigmahat_plus = HBMP_sigma_plus(sigmahat, n, delta/2, maxiters)
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n) 
        bentkus_mu = bentkus_plus(mu, muhat, n)
        bennett_mu = bennett_plus(mu, sigmahat_plus, muhat, n, num_grid)
        return min(hoeffding_mu, bentkus_mu, bennett_mu) - np.log(delta / 2)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)

def HB_mu_plus(muhat, sigmahat, n, delta, num_grid, maxiters):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n) 
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)

### UCB of mean via Empirical Bennett
def empirical_bennett_mu_plus(muhat, sigmahat, n, delta, num_grid, maxiters):
    sigmahat_plus = HBMP_sigma_plus(sigmahat, n, delta/2, maxiters)
    def _tailprob(mu):
        return bennett_plus(mu, sigmahat_plus, muhat, n, num_grid) - np.log(delta / 2)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)

### UCB of mean via Bentkus
def bentkus_mu_plus(muhat, sigmahat, n, delta, num_grid, maxiters):
    def _tailprob(mu):
        return bentkus_plus(mu, muhat, n) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)

### UCB of mean via Bentkus
def hoeffding_mu_plus(muhat, sigmahat, n, delta, num_grid, maxiters):
    def _tailprob(mu):
        return hoeffding_plus(mu, muhat, n) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)

### UCB of mean via Bentkus
def hoeffding_naive_mu_plus(muhat, sigmahat, n, delta, num_grid, maxiters):
    def _tailprob(mu):
        return hoeffding_naive(mu, muhat, n) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)

def pinelis_utev_mu_plus(muhat, n, delta, cv, maxiters):
    def _tailprob(mu):
        return pinelis_utev(mu, muhat, n, cv) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)

def WSR_mu_plus(x, delta, maxiters): # this one is different.
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1,n+1)))
    sigma2hat = (np.cumsum((x - muhat)**2) + 0.25) / (1 + np.array(range(1,n+1))) 
    sigma2hat[1:] = sigma2hat[:-1]
    sigma2hat[0] = 0.25
    nu = np.minimum(np.sqrt(2 * np.log( 1 / delta ) / n / sigma2hat), 1)
    def _Kn(mu):
        return np.max(np.cumsum(np.log(1 - nu * (x - mu)))) + np.log(delta)
    if _Kn(1) < 0:
        return 1
    return brentq(_Kn, 1e-10, 1-1e-10, maxiter=maxiters)


if __name__ == "__main__":
    print(empirical_bennett_mu_plus(0.1, 0.01, 10000, 0.1, 100, 1000))
    print(WSR_mu_plus(0.1+np.random.random(size=(1000,))/100, 0.01, 1000))
