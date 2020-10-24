## Some auxiliary functions
safe_min <- function(x){
    if (any(is.na(x))){
        -Inf
    } else {
        min(x)
    }
}

h1 <- function(y, mu){
    y * log(y / mu) + (1 - y) * log((1 - y) / (1 - mu))
}

h2 <- function(y){
    (1 + y) * log(1 + y) - y
}

## Log upper-tail inequalities of mean
hoeffding_plus <- function(mu, x, n){
    -n * h1(pmin(mu, x), mu)
}

bennett_plus <- function(mu, sigma, x, n){
    gamma <- seq(0, mu * 0.99, length.out = 100)
    v <- (sigma^2 + gamma^2)
    b <- mu - gamma
    t <- pmax(mu - x, 0)
    res <- v / b^2 * h2(b * t / v)
    -n * max(res)
}

bentkus_plus <- function(mu, x, n){
    log(pbinom(ceiling(n * x), n, mu, lower.tail = TRUE)) + 1
}

## Log upper-tail inequalities of empirical variance
hoeffding_var <- function(sigma2, x, n){
    m <- floor(n / 2)
    -m * h1(pmin(sigma2, x), sigma2)
}

bentkus_var <- function(sigma2, x, n){
    m <- floor(n / 2)
    log(pbinom(ceiling(m * x), m, sigma2, lower.tail = TRUE)) + 1
}

maurer_pontil_var <- function(sigma2, x, n){
    -(n - 1) / 2 / sigma2 * pmax(sigma2 - x, 0)^2
}

## Upper confidence bound of mean via Hoeffding-Bentkus-Maurer-Pontil inequalities
HBMP_sigma_plus <- function(sigmahat, n, alpha){
    sigma2hat <- sigmahat^2
    tailprob <- function(sigma2){
        hoeffding_sigma <- hoeffding_var(sigma2, sigma2hat, n)
        bentkus_sigma <- bentkus_var(sigma2, sigma2hat, n)
        maurer_pontil_sigma <- maurer_pontil_var(sigma2, sigma2hat, n)
        min(hoeffding_sigma, bentkus_sigma, maurer_pontil_sigma) - log(alpha)
    }
    if (tailprob(0.25) > 0){
        0.5
    } else {
        sqrt(uniroot(tailprob, c(sigma2hat, 0.25))$root)
    }
}

#' Upper confidence bound of mean via Hoeffding-Bentkus-Empirical Bennett inequalities (w/ alpha / 2 on the mean and alpha / 2 on the variance)
#'
#' @param muhat empirical average
#' @param sigmahat empirical estimate of standard deviation
#' @param n sample size
#' @param alpha confidence level
#' 
HBB_mu_plus <- function(muhat, sigmahat, n, alpha){
    sigmahat_plus <- HBMP_sigma_plus(sigmahat, n, alpha / 2)
    tailprob <- function(mu){
        hoeffding_mu <- hoeffding_plus(mu, muhat, n)
        bentkus_mu <- bentkus_plus(mu, muhat, n)
        bennett_mu <- bennett_plus(mu, sigmahat_plus, muhat, n)
        min(hoeffding_mu, bentkus_mu, bennett_mu) - log(alpha / 2)
    }
    if (tailprob(1 - 1e-10) > 0){
        1
    } else {
        uniroot(tailprob, c(muhat, 1 - 1e-10))$root
    }
}

#' Upper confidence bound of mean via Hoeffding-Bentkus-Empirical Bennett inequalities (w/ alpha / 2 on the mean and alpha / 2 on the variance)
#'
#' @param muhat empirical average
#' @param sigmahat empirical estimate of standard deviation
#' @param n sample size
#' @param alpha confidence level
#' 
Emp_bennett_mu_plus <- function(muhat, sigmahat, n, alpha){
    sigmahat_plus <- HBMP_sigma_plus(sigmahat, n, alpha / 2)
    tailprob <- function(mu){
        bennett_mu <- bennett_plus(mu, sigmahat_plus, muhat, n)
        bennett_mu - log(alpha / 2)
    }
    if (tailprob(1 - 1e-10) > 0){
        1
    } else {
        uniroot(tailprob, c(muhat, 1 - 1e-10))$root
    }
}