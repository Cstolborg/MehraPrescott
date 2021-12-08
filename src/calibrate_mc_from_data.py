from collections import namedtuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.rouwenhorst import rouwenhorst
from src.markov_chain import MarkovChain


class CalibrateMcChainFromData:
    """
    Calibrates an n-state Markov Chain from growth data.
    Takes a pandas DataFrame with index_col='year' and
    growth column named 'growth'.
    """

    def __init__(self, data, n_states=2, start_year=None):
        if not start_year is None:
            data = data.loc[start_year:]

        self.n_states = n_states
        self.data = data
        self.g = self.data['growth'].iloc[1:]
        self.g1 = self.data['growth'].shift(1).dropna()

        self.called = False  # Call method not used yet

    def __call__(self, summary=False):
        # Fit AR(1) model and compute unconditional moments
        self.ar_result, self.ar_summary = self.fit_ar(self.g, self.g1, summary=summary)
        self.mu, self.rho, self.sigma = self.get_uncond_moments_ar()

        # Apply rouwenhorst method to calibrate Markov Chain from AR(1) moments
        x, Pi = rouwenhorst(n=self.n_states, mu=self.mu, sigma=self.sigma, rho=self.rho)  # state-vector and TPM
        self.mc = MarkovChain(Pi, x)
        self.called = True

        return MarkovChain(Pi, x)

    def fit_ar(self, x, x_lagged, summary=False):
        X = sm.add_constant(x_lagged)
        model = sm.OLS(x, X)
        res = model.fit()

        const, rho = res.params  # Const + AR(1) param
        sigma = np.sqrt(res.scale)

        if summary: print(res.summary())

        ArResult = namedtuple('ArResult', 'const, rho, sigma')
        return ArResult(const, rho, sigma), res.summary()

    def get_uncond_moments_ar(self):
        const, rho, sigma = self.ar_result  # Extract params

        ar_mean = const / (1 - rho) + 1
        ar_std = np.sqrt(sigma ** 2 / (1 - rho ** 2))
        ar_rho = rho  # rho * ar_std**2
        return ar_mean, ar_rho, ar_std

    def summary(self):
        if not self.called: self()  # Call method
        # Summarize results
        const, rho, sigma = self.ar_result
        ar_mean, ar_rho, ar_std = (self.mu, self.rho, self.sigma)
        mc_params = self.mc.uncond_moments()

        summ = pd.DataFrame([[const, rho, sigma],
                             [ar_mean, ar_rho, ar_std],
                             mc_params
                             ],
                            columns=['Mean', 'Rho', 'Std'],
                            index=['AR coef', 'AR moments', "Markov Chain"])

        out = {'Z': self.mc.x,
               'P': self.mc.Pi,
               'stationary dist': self.mc.pi_bar,
               'mean': mc_params[0],
               'std': mc_params[2],
               'rho': mc_params[1]}

        return out, summ


if __name__ == '__main__':
    data = pd.read_excel("../data/PCE growth data.xlsx", index_col="year")

    cal_chain =  CalibrateMcChainFromData(data, n_states=2)
    cal_chain.summary()