import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from markov_chain import MarkovChain
from calibrate_mc_from_data import CalibrateMcChainFromData

class MehraEconomy:
    """
    Class for computing asset prices in the Lucas Tree Economy
    used by Mehra and Prescott (1985)

    """

    def __init__(self, mc: MarkovChain, beta=np.exp(-0.02), gamma=2):
        self.mc = mc
        self.x, self.Pi, self.pi_bar = mc.x, mc.Pi, mc.pi_bar
        self.n_states = len(self.x)

        self.beta = beta
        self.gamma = gamma

        self.bonds = MehraBonds(self.mc, beta=beta, gamma=gamma)
        self.stocks = MehraStocks(self.mc, beta=beta, gamma=gamma)
        self.bonds(beta, gamma)  # Run __call__ method
        self.stocks(beta, gamma)

    def __call__(self, beta, gamma):
        # Init bonds and equity prices - notice difference between self for MC and local variables for beta+gamma
        self.bonds = MehraBonds(self.mc, beta=beta, gamma=gamma)
        self.stocks = MehraStocks(self.mc, beta=beta, gamma=gamma)

        self.bonds(beta, gamma)  # Run __call__ method
        self.stocks(beta, gamma)

    def calibrate_beta_to_rf(self, target_rf=0.05):
        def target_func(beta):
            Rf_i = 1 / self.bonds.prices(beta) - 1
            Rf = self.pi_bar @ Rf_i
            return Rf - target_rf

        self.beta = opt.fsolve(target_func, x0=np.exp(-0.02))[0]

    def excess_ret(self):
        self(beta=self.beta, gamma=self.gamma)
        self.equity_ret_ = self.stocks.ret_
        self.bond_ret_ = self.bonds.ret_

        self.excess_ret_ = self.equity_ret_ - self.bond_ret_
        return self.excess_ret_

    def num_experiment(self, N=100, rf_bound=(0.01, 0.04)):
        lower, upper = rf_bound
        rfs = np.linspace(lower, upper, num=N)
        gammas1 = np.linspace(0, 10, num=N)

        gammas = []
        betas = []
        re = []
        rf = []
        ER = []
        for r in rfs:
            for gamma in gammas1:
                self.calibrate_beta_to_rf(target_rf=r)
                self(self.beta, gamma)
                if self.beta >= 1. or self.beta <= 0.95:
                    continue
                self.gamma = gamma
                ER.append(self.excess_ret())
                betas.append(self.beta)
                re.append(self.equity_ret_)
                rf.append(self.bond_ret_)
                gammas.append(gamma)

        return ER, rf, re, betas, gammas

class MehraBonds:
    """
    Class for computing bond prices in a Mehra-Prescott economy
    """

    def __init__(self, mc:MarkovChain, beta, gamma):
        self.mc = mc
        self.x, self.Pi, self.pi_bar = mc.x, mc.Pi, mc.pi_bar
        self.n_states = len(self.x)
        self.beta = beta
        self.gamma = gamma

    def __call__(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma

        self.ret()  # Calls all other methods

    def prices(self, beta=None):
        if beta is None: beta = self.beta
        # Compute risk-free return from Markov Chain
        prices = np.zeros(self.n_states)
        for i in range(self.n_states):
            for j in range(self.n_states):
                prices[i] += beta * self.Pi[i, j] * self.x[j]**(-self.gamma)

        self.prices_ = prices
        return prices

    def rets(self):
        prices = self.prices()
        rets = 1/prices - 1  # Net return
        self.rets_ = rets
        return rets

    def ret(self):
        self.ret_ = self.pi_bar @ self.rets()
        return self.ret_

class MehraStocks:
    """
    Class for computing stock prices and returns in a Mehra-Prescott economy
    """

    def __init__(self, mc:MarkovChain, beta, gamma):
        self.mc = mc
        self.x, self.Pi, self.pi_bar = mc.x, mc.Pi, mc.pi_bar
        self.n_states = len(self.x)
        self.beta = beta
        self.gamma = gamma

    def __call__(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma

        self.ret()  # Calls all other methods

    def prices(self):
        A = self.beta * self.x**(1-self.gamma) * self.Pi
        b = A.sum(axis=1)

        I = np.identity(2)
        S = np.linalg.inv(I - A) @ b

        self.prices_ = S
        return S

    def realized_rets(self):
        re = np.zeros((2, 2))
        S = self.prices()
        for i in range(2):
            for j in range(2):
                re[i, j] = self.x[j] * (S[j] + 1) / S[i] - 1

        self.realized_rets_ = re
        return re

    def rets(self):
        real_rets = self.realized_rets()  # NxN matrix
        self.rets_ = (real_rets * self.Pi).sum(axis=1)
        return self.rets_

    def ret(self):
        self.ret_ = self.pi_bar @ self.rets()
        return self.ret_

    
