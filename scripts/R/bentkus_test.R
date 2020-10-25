# This document tests the benkus bound in a the bernoulli case. 
# Here, the bentkus bound should be exact (up to discretization), except that the probability is off by a factor of e.
# We simulate bernoulli variables, compute their mean, and see how often the bentkus bound miscovers.

#tail bound according to bentkus
bentkus_plus <- function(mu, x, n){
  log(pbinom(floor(n * x), n, mu, lower.tail = TRUE)) + 1
}

#find ucb for mu from muhat using the bentkus bound
bentkus_ucb <- function(muhat, delta, n_cal) {
  tailprob <- function(mu) {
    bentkus_plus(mu, muhat, n_cal) - log(delta)
  } 
  uniroot(tailprob, c(muhat, 1 - 1e-10), maxiter = 5000, tol = .00001)$root
}

#find the threshold below which mu_hat will result in a miscoverage
bentkus_lcb <- function(mu, delta, n_cal) {
  tailprob <- function(muhat) {
    bentkus_plus(mu, muhat, n_cal) - log(delta)
  } 
  uniroot(tailprob, c(1e-10, mu), maxiter = 5000, tol = .00001)$root
}

#simulation setting
n_cal <- 10000
n_reps <- 1000000
mus <- c(.05, .1, .2)
deltas <- c(.2, .1, .05, .01)

#check that probability in the lower tail is at the desired level for a single case:
delta <- .1
muhat <- .1
ucb <- bentkus_ucb(muhat, delta, n_cal)
x <- rbinom(n_reps*10, n_cal, ucb) / n_cal
mean(x <= muhat) * exp(1) / delta #simulation / theory (should be near 1)


#check coverage for bernoulli across simulation settings
for(mu in mus) {
  for(delta in deltas) {
    print(paste0("mu: ", mu, "  delta: ", delta))
    
    #thresh <- bentkus_lcb(mu, delta, n_cal) #find the value below which bentkus miscovers
    
    #find the value below which bentkus miscovers by inverting the ucb (this version has more numerical error than the above)
    thresh <- uniroot(function(muhat){bentkus_ucb(muhat, delta, n_cal) - mu}, c(1e-10, mu), maxiter = 5000, tol = .00001)$root 
    
    x <- rbinom(n_reps, n_cal, mu) / n_cal #simulate mean of binomial
    print(paste0("empirical miscoverage / theory miscoverage: ", (mean(x <= thresh) / (delta / exp(1)))))
  }
}

