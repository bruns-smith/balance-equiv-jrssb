import numpy as np
import scipy.stats
from scipy.stats import special_ortho_group

def setup_source_dgp(scaler=1, min_eig=1e-6, max_eig=2, n=2000, d=50):

    # coefficients of outcome model
    beta_zero = np.abs(np.random.normal(0,1,size=d))
    beta_zero /= np.linalg.norm(beta_zero)
    beta_zero *= scaler

    # source covariate population covariance matrix:
    base = np.eye(d)
    eigs = np.linspace(max_eig, min_eig, d)
    train_cov = base.T@np.diag(eigs)@base
    
    return beta_zero, train_cov

def setup_curved_source_dgp(scaler=1, min_eig=1e-6, max_eig=2, curve=5000, n=2000, d=50):

    # coefficients of outcome model
    beta_zero = np.abs(np.random.normal(0,1,size=d))
    beta_zero /= np.linalg.norm(beta_zero)
    beta_zero *= scaler

    # source covariate population covariance matrix:
    base = special_ortho_group.rvs(d) #np.eye(d)
    eigs = (np.linspace(max_eig**(1/curve), min_eig**(1/curve), d))**curve
    train_cov = base.T@np.diag(eigs)@base

    return beta_zero, train_cov

def draw_X(train_cov, n):
    # fixed design
    d = train_cov.shape[0]
    source_dist = scipy.stats.multivariate_normal(mean=np.zeros(d), cov=train_cov, allow_singular=True)
    X = source_dist.rvs(size=n)
    sigma_hat = (X.T @ X)/n
    return X, sigma_hat

def mean_shift_tests(d, scaler=1):
    test_mean = np.ones(d)
    test_mean *= scaler
    test_mean_outer = np.outer(test_mean, test_mean)
    return test_mean_outer, test_mean