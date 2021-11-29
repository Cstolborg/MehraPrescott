import os

import numpy as np
import pandas as pd
from math import erfc, sqrt
from scipy import stats
import statsmodels.api as sm

from utils import rouwenhorst

pd.set_option('display.max_columns', 500)


def get_data_growth_lags(path, start_year=None):
    data = pd.read_excel(path, index_col="year")

    if not start_year is None:
        data = data.loc[start_year:]

    growth = data['growth']
    growth_lag1 = growth.shift(1).dropna()
    growth = growth.iloc[1:]

    return data, growth, growth_lag1

def fit_ar(x, x_lagged, components=1, summary=False):
    if not components == 1:
        x_lagged2 = x_lagged.shift(1).dropna()

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

def get_mc_params(Z, P):
    # Compute unconditional probabilities (ergodic distribution):
    eig_val, eig_vec = np.linalg.eig(P.T)
    eig_vec = eig_vec[:, np.argmax(eig_val)]
    pi_bar = eig_vec / eig_vec.sum()

    mean = pi_bar @ Z
    std = np.sqrt(pi_bar @ Z ** 2 - (pi_bar @ Z) ** 2)
    rho = np.trace(P) - 1

    return pi_bar, mean, rho, std

def run(n, path, start_year=None):
    data, growth, growth_lag1 = get_data_growth_lags(path, start_year=start_year)

    const, rho, sigma = fit_ar(growth, growth_lag1, summary=False)

    ar_mean, ar_rho, ar_std = get_uncond_moments_ar(const, rho, sigma)

    # Calibrate 2-state chain
    Z, P = rouwenhorst(n=n, mu=ar_mean, sigma=ar_std, rho=rho)  # state-vector and TPM
    mc_params = get_mc_params(Z, P)  # pi, mean, std, rho

    # Summarize results
    summ = pd.DataFrame([[const, rho, sigma],
                         [ar_mean, ar_rho, ar_std],
                         mc_params[1:]],
                        columns=['mean', 'rho', 'std'],
                        index=['regression', 'ar_moments', "mc"])

    out = {'Z': Z,
           'P': P,
           'stationary dist': mc_params[0],
           'mean': mc_params[1],
           'std': mc_params[3],
           'rho': mc_params[2]}

    return out, summ

if __name__ == '__main__':
    # Get data and fit an AR model
    path_nipa = "./data/PCE growth data.xlsx"
    path_mehra = "./data/Shiller data extended.xlsx"

    run(n=2, path=path_nipa, start_year=1950)
    run(n=10, path=path_mehra)

    """
    data, growth, growth_lag1 = get_data_growth_lags(path)

    const, rho, sigma = fit_ar(growth, growth_lag1, summary=False)

    ar_mean, ar_rho, ar_std = get_uncond_moments_ar(const, rho, sigma)

    # Calibrate 2-state chain
    Z, P = rouwenhorst(n=2, mu=ar_mean, sigma=ar_std, rho=rho)  # state-vector and TPM
    mc2_params = get_mc_params(Z, P)  # pi, mean, std, rho

    # Calibrate 10-state chain
    Z, P = rouwenhorst(n=10, mu=ar_mean, sigma=ar_std, rho=rho)  # state-vector and TPM
    mc10_params = get_mc_params(Z, P)  # pi, mean, std, rho

    # Check Mehra's data

    mehra_data = pd.read_csv("data/Shiller data extended.csv", index_col='Unnamed: 9')['C_growth'].dropna()
    mehra_g = mehra_data.loc[1890:1978]
    mehra_g1 = mehra_g.shift(1).dropna()
    mehra_g = mehra_g.iloc[1:]

    const_m, rho_m, sigma_m = fit_ar(mehra_g, mehra_g1, summary=True)

    ar_mean_m, ar_rho_m, ar_std_m = get_uncond_moments_ar(const_m, rho_m, sigma_m)

    # Calibrate 2-state chain
    Z, P = rouwenhorst(n=2, mu=ar_mean_m, sigma=ar_std_m, rho=rho_m)  # state-vector and TPM
    mc_mehra_params = get_mc_params(Z, P)  # pi, mean, std, rho


    Z, P = rouwenhorst(n=2, mu=1.018, sigma=0.036, rho=-.14)  # state-vector and TPM
    mehra_params = get_mc_params(Z, P)  # pi, mean, std, rho



    # Summarize results
    summ = pd.DataFrame([[const, rho, sigma],
                         [ar_mean, ar_rho, ar_std],
                         mc2_params[1:],
                         mc10_params[1:]],
                        columns=['mean', 'rho', 'std'],
                        index=['regression', 'ar_moments', "mc2", "mc10"])

    print(summ)
    """