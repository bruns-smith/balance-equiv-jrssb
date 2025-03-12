import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso

import cvxpy as cp

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1, BFGS
from scipy.optimize import minimize

import scipy.stats
from scipy.stats import special_ortho_group

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, KFold

plt.rcParams.update({'font.size': 16})


class RidgeAnalysis:

    def __init__(self, beta_zero, sigma_hat, train_cov, X):
        self.beta_zero = beta_zero
        self.sigma_hat = sigma_hat
        self.train_cov = train_cov

        self.X = X
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]

        self.evals, self.evecs = np.linalg.eigh(self.sigma_hat)

    def calc_gamma(self, lam, delta):
        gamma = lam * delta / (self.evals + lam + delta)
        return gamma

    def ridge_bias(self, lam, reg_inv_cov, target_mat ):
        return lam**2 * self.beta_zero @ (reg_inv_cov @ target_mat @ reg_inv_cov) @ self.beta_zero

    def ridge_variance(self, reg_inv_cov, target_mat, sigma2_y ):
        var_mat = self.sigma_hat @ reg_inv_cov @ target_mat @ reg_inv_cov
        return (sigma2_y/self.n) * np.trace(var_mat)

    def general_ridge_bias(self, D, reg_inv_cov, target_mat ):
        return self.beta_zero @ (reg_inv_cov @ D @ target_mat @ D @ reg_inv_cov) @ self.beta_zero

    def general_ridge_variance(self, reg_inv_cov, target_mat, sigma2_y ):
        var_mat = self.sigma_hat @ reg_inv_cov @ target_mat @ reg_inv_cov
        return (sigma2_y/self.n) * np.trace(var_mat)  

    def gen_iid_objective(self, sigma2_y):
        def iid_objective(lam):
            reg_inv_cov = self.evecs @ self.pseudo_inv_diag(self.evals + lam, 1e-12) @ self.evecs.T
            bias2 = self.ridge_bias( lam, reg_inv_cov, self.train_cov )
            variance = self.ridge_variance( reg_inv_cov, self.train_cov, sigma2_y )
            return bias2+variance
        return iid_objective

    def pseudo_inv_diag(self, reg_evals, eps):
        ind = reg_evals > eps
        res = np.zeros(reg_evals.shape[0])
        res[ind] = 1/(reg_evals[ind])
        return np.diag(res)

    def gen_mean_objective(self, test_mean_outer, sigma2_y, opt_lambda, obj_scale):
        def mean_objective(delta):
            gamma = self.calc_gamma(opt_lambda, delta)
            reg_inv_cov = self.evecs @ self.pseudo_inv_diag(self.evals + gamma, 1e-12) @ self.evecs.T
            reg_mat = self.evecs @ np.diag(gamma) @ self.evecs.T
            
            bias2 = self.general_ridge_bias( reg_mat, reg_inv_cov, test_mean_outer ) / obj_scale
            variance = self.general_ridge_variance( reg_inv_cov, test_mean_outer, sigma2_y ) / obj_scale
            return bias2+variance
        return mean_objective
    
    def solve_optimization(sef, obj):
        res = scipy.optimize.minimize(obj, x0=0.01, bounds=[(0,100000)], tol=1e-12)
        # res = scipy.optimize.shgo(obj, bounds=[(0,100000)], n=10000, iters=100)
        #res = scipy.optimize.basinhopping(obj, x0=0.01, niter=200)
        return res.x[0], res.fun
    
    def estimate_for_delta(self, opt_lambda, delta, test_mean, y):
        gamma = self.calc_gamma(opt_lambda, delta)
        reg_inv_cov = self.evecs @ np.diag(1/(self.evals + gamma)) @ self.evecs.T
        beta = reg_inv_cov @ self.X.T @ y / self.n
        return beta @ test_mean
    
    def calc_mc_sq_err(self, y, lam, delta, test_mean):
        gamma = self.calc_gamma(lam, delta)
        reg_inv_cov = self.evecs @ np.diag(1/(self.evals + gamma)) @ self.evecs.T
        beta = reg_inv_cov @ self.X.T @ y / self.n
        point_estimate = beta @ test_mean
        truth = self.beta_zero @ test_mean
        return (point_estimate - truth)**2