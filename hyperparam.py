import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import minimize

def autodml_loss(rho, Xp, Xqb):
    m = Xp.shape[0]
    return -2 * Xqb@rho + rho@(Xp.T@Xp/m)@rho

def calc_imbal(theta, Xp, Xq):
    m = Xp.shape[0]
    Xqhat = ((Xp.T@Xp)/m) @ theta
    imbal = np.linalg.norm( Xqhat - Xq, np.inf)
    return imbal

def pseudo_inv_diag(reg_evals, eps):
    ind = reg_evals > eps
    res = np.zeros(reg_evals.shape[0])
    res[ind] = 1/(reg_evals[ind])
    return np.diag(res)

def cross_val_autodml_solver(test_mean, X, repeats=1, plot=False, plot_deltas=None, seed=None, ks=10):

    cv_deltas = []
    for _ in range(repeats):
        kf = KFold(n_splits=ks, shuffle=True, random_state=seed) # 
        all_tests = []
        all_treval = []
        all_trevec = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            trainMean = X[train_index, :].mean(axis=0)
            Xtrain = X[train_index, :] - trainMean
            Xtest = X[test_index, :] - trainMean

            nt = Xtrain.shape[0]
            
            treval, trevec = np.linalg.eigh( Xtrain.T@Xtrain/nt )
            all_treval.append(treval)
            all_trevec.append(trevec)
            all_tests.append(Xtest)
            
        def cross_val_loss_calc(delta):
            loss = 0
            for treval,trevec,Xtest in zip(all_treval, all_trevec, all_tests):
                theta = test_mean @ trevec @ pseudo_inv_diag(treval+delta, 1e-12) @ trevec.T
                loss += autodml_loss(theta, Xtest, test_mean)/ks
            return loss
        
        if plot:
            return [cross_val_loss_calc(delta) for delta in plot_deltas]
        
        res = minimize(cross_val_loss_calc, x0=np.random.random(), bounds=[(0,100000)], tol=1e-20)
        
        cv_deltas.append(res.x[0])
    return cv_deltas

def cross_val_bal_solver(test_mean, X, repeats=5, plot=False, plot_deltas=None, seed=None, ks=2):

    cv_deltas = []
    for _ in range(repeats):
        ks = 2
        kf = KFold(n_splits=ks, shuffle=True, random_state=seed)
        all_tests = []
        all_treval = []
        all_trevec = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            trainMean = X[train_index, :].mean(axis=0)
            Xtrain = X[train_index, :] - trainMean
            Xtest = X[test_index, :] - trainMean

            nt = Xtrain.shape[0]
            
            treval, trevec = np.linalg.eigh( Xtrain.T@Xtrain/nt )
            all_treval.append(treval)
            all_trevec.append(trevec)
            all_tests.append(Xtest)
            
        def cv_bal_calc(delta, all_treval, all_trevec, all_tests):
            imbal = 0
            for treval,trevec,Xtest in zip(all_treval, all_trevec, all_tests):
                theta = test_mean @ trevec @ pseudo_inv_diag(treval+delta, 1e-12) @ trevec.T
                Xqhat = theta @ (Xtest.T @ Xtest / Xtest.shape[0])
                imbal += ((Xqhat - test_mean)**2).sum()/ks
            return imbal

        def obj(delta):
            return cv_bal_calc(delta, all_treval, all_trevec, all_tests)
        
        if plot:
            return [obj(delta) for delta in plot_deltas]
    
        res = minimize(obj, x0=np.random.random(), bounds=[(0,100000)], tol=1e-12)

        # if res.x[0] < 0:
        #     deltas = np.linspace(0,0.1, 1000)
        #     print(ts)
        #     return deltas, [obj(delta) for delta in deltas]

        cv_deltas.append(res.x[0])
        
    return cv_deltas

def cross_val_ridge_solver(y, X, repeats=5):

    lams = []
    for _ in range(repeats):
        ks = 5
        kf = KFold(n_splits=ks, shuffle=True)
        all_tests = []
        all_trains = []
        all_treval = []
        all_trevec = []
        all_train_y = []
        all_test_y = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            trainMean  = X[train_index, :].mean(axis=0)
            Xtrain = X[train_index, :] - trainMean
            Xtest = X[test_index, :] - trainMean

            ymean = y[train_index].mean()
            ytrain = y[train_index] - ymean
            ytest = y[test_index] - ymean
            
            treval, trevec = np.linalg.eigh( Xtrain.T@Xtrain/Xtrain.shape[0] )
            all_treval.append(treval)
            all_trevec.append(trevec)
            all_tests.append(Xtest)
            all_trains.append(Xtrain)
            all_train_y.append(ytrain)
            all_test_y.append(ytest)
            
        def cv_calc(lam):
            mse = 0
            for treval,trevec,Xtest,Xtrain,ytest,ytrain in zip(all_treval, all_trevec, all_tests, all_trains, all_test_y, all_train_y):
                reg_inv_cov = trevec @ pseudo_inv_diag(treval+lam, 1e-12) @ trevec.T
                beta = reg_inv_cov @ Xtrain.T @ ytrain / Xtrain.shape[0]
                yhat = Xtest @ beta
                err = ((yhat - ytest)**2).mean()
                mse += err/ks
            return mse
        
        res = minimize(cv_calc, x0=0.01, bounds=[(0,100000)], tol=1e-12)

        lams.append(res.x)
        
    return np.quantile(lams, 0.5)

def cross_val_bal_grid(test_mean, X, grid, repeats=5, plot=False, seed=None):

    cv_deltas = []
    for _ in range(repeats):
        ks = 2
        kf = KFold(n_splits=ks, shuffle=True, random_state=seed)
        all_tests = []
        all_treval = []
        all_trevec = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            trainMean = X[train_index, :].mean(axis=0)
            Xtrain = X[train_index, :] - trainMean
            Xtest = X[test_index, :] - trainMean

            nt = Xtrain.shape[0]
            
            treval, trevec = np.linalg.eigh( Xtrain.T@Xtrain/nt )
            all_treval.append(treval)
            all_trevec.append(trevec)
            all_tests.append(Xtest)
            
        def cv_bal_calc(delta, all_treval, all_trevec, all_tests):
            imbal = 0
            for treval,trevec,Xtest in zip(all_treval, all_trevec, all_tests):
                theta = test_mean @ trevec @ np.diag( 1/(treval+delta) ) @ trevec.T
                Xqhat = theta @ (Xtest.T @ Xtest / Xtest.shape[0])
                imbal += ((Xqhat - test_mean)**2).sum()/ks
            return imbal

        def obj(delta):
            return cv_bal_calc(delta, all_treval, all_trevec, all_tests)
        
        deltas = grid
        imbals = np.array([obj(delta) for delta in deltas])

        if plot:
            return deltas, imbals
    
        res = deltas[np.argmin(imbals)]

        cv_deltas.append(res)
        
    return cv_deltas

def cross_val_autodml_grid(test_mean, X, grid, repeats=1, plot=False, seed=None):

    cv_deltas = []
    for _ in range(repeats):

        ks = 10
        kf = KFold(n_splits=ks, shuffle=True, random_state=seed) # 
        all_tests = []
        all_treval = []
        all_trevec = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            trainMean = X[train_index, :].mean(axis=0)
            Xtrain = X[train_index, :] - trainMean
            Xtest = X[test_index, :] - trainMean

            nt = Xtrain.shape[0]
            
            treval, trevec = np.linalg.eigh( Xtrain.T@Xtrain/nt )
            all_treval.append(treval)
            all_trevec.append(trevec)
            all_tests.append(Xtest)
            
        def cross_val_loss_calc(delta):
            loss = 0
            for treval,trevec,Xtest in zip(all_treval, all_trevec, all_tests):
                theta = test_mean @ trevec @ np.diag( 1/(treval+delta) ) @ trevec.T
                loss += autodml_loss(theta, Xtest, test_mean)/ks
            return loss
        
        deltas = grid
        imbals = np.array([cross_val_loss_calc(delta) for delta in deltas])

        if plot:
            return deltas, imbals
    
        res = deltas[np.argmin(imbals)]
        
        cv_deltas.append(res)

    return cv_deltas