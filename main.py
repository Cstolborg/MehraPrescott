import os

import numpy as np
import pandas as pd
from math import erfc, sqrt
from scipy import stats
import statsmodels.api as sm

from utils import rouwenhorst

pd.set_option('display.max_columns', 500)

def get_data_growth_lags(path):
    data = pd.read_excel(path, index_col="Year")

    growth = data['Growth']
    growth_lag1 = growth.shift(1).dropna()
    growth = growth.iloc[1:]

    return data, growth, growth_lag1

def fit_ar(x, x_lagged, summary=False):
    X = sm.add_constant(x_lagged)
    model = sm.OLS(x, X)
    res = model.fit()

    const, rho = res.params  # Const + AR(1) param
    sigma = np.sqrt(res.scale)

    if summary: print(res.summary())

    return const, rho, sigma

def get_uncond_moments_ar(const, rho, sigma):
    ar_mean = const / (1 - rho) + 1
    ar_std = np.sqrt(sigma ** 2 / (1 - rho ** 2))
    ar_rho = rho  # rho * ar_std**2

    return ar_mean, ar_rho, ar_std

def mc_params(Z, P):
    # Compute unconditional probabilities (ergodic distribution):
    eig_val, eig_vec = np.linalg.eig(P.T)
    eig_vec = eig_vec[:, np.argmax(eig_val)]
    pi_bar = eig_vec / eig_vec.sum()

    mean = pi_bar @ Z
    std = np.sqrt(pi_bar @ Z ** 2 - (pi_bar @ Z) ** 2)
    rho = np.trace(P) - 1

    return pi_bar, mean, rho, std



if __name__ == '__main__':
    path = "./PCE growth data.xlsx"
    data, growth, growth_lag1 = get_data_growth_lags(path)

    const, rho, sigma = fit_ar(growth, growth_lag1, summary=False)

    ar_mean, ar_rho, ar_std = get_uncond_moments_ar(const, rho, sigma)

    # Calibrate 2-state chain
    Z, P = rouwenhorst(n=2, mu=ar_mean, sigma=ar_std, rho=rho)  # state-vector and TPM
    mc2_params = mc_params(Z, P)  # pi, mean, std, rho

    # Calibrate 10-state chain
    Z, P = rouwenhorst(n=10, mu=ar_mean, sigma=ar_std, rho=rho)  # state-vector and TPM
    mc10_params = mc_params(Z, P)  # pi, mean, std, rho

    # Check Mehra's data
    Z, P = rouwenhorst(n=2, mu=1.018, sigma=0.036, rho=-.14)  # state-vector and TPM
    mehra_params = mc_params(Z, P)  # pi, mean, std, rho



    # Summarize results
    summ = pd.DataFrame([[const, rho, sigma],
                         [ar_mean, ar_rho, ar_std],
                         mc2_params[1:],
                         mc10_params[1:]],
                        columns=['mean', 'rho', 'std'],
                        index=['regression', 'ar_moments', "mc2", "mc10"])

    print(summ)