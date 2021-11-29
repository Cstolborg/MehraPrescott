import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from src.markov_chain import MarkovChain
from src.calibrate_mc_from_data import CalibrateMcChainFromData

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
        self.excess_ret_ = self.stocks.ret_ - self.bonds.ret_

    def calibrate_beta_to_rf(self, target_rf=0.05):
        """
        Given target risk-free rate choose a beta that matches
        """
        def target_func(beta):
            Rf_i = 1 / self.bonds.prices(beta) - 1
            Rf = self.pi_bar @ Rf_i
            return Rf - target_rf

        self.beta = opt.fsolve(target_func, x0=np.exp(-0.02))[0]

    def excess_ret(self):
        """ Unconditional excess returns """
        self(beta=self.beta, gamma=self.gamma)

        self.excess_ret_ = self.stocks.ret_ - self.bonds.ret_
        return self.excess_ret_

    def num_experiment(self, N=100, rf_bound=(0.01, 0.04)):
        """
        Compares which excess returns pairs of gamma and beta can produce
        """
        rf_lower, rf_upper = rf_bound
        betas1 = np.linspace(0.95, 0.999, num=N)
        gammas1 = np.linspace(0.1, 10, num=N)

        gammas = []
        betas = []
        re = []
        rf = []
        ER = []
        for beta in betas1:
            for gamma in gammas1:
                #self.calibrate_beta_to_rf(target_rf=r)
                self(beta, gamma)
                if not rf_lower< self.bonds.ret_ <= rf_upper:
                    continue
                self.gamma = gamma
                ER.append(self.excess_ret_)
                betas.append(beta)
                re.append(self.stocks.ret_)
                rf.append(self.bonds.ret_)
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
        """ Conditional bond prices in each state - outputs (n_states x 1) array """
        if beta is None: beta = self.beta
        # Compute risk-free return from Markov Chain
        prices = np.zeros(self.n_states)
        for i in range(self.n_states):
            for j in range(self.n_states):
                prices[i] += beta * self.Pi[i, j] * self.x[j]**(-self.gamma)

        self.prices_ = prices
        return prices

    def rets(self):
        """ Conditional returns in each state """
        prices = self.prices()
        rets = 1/prices - 1  # Net return
        self.rets_ = rets
        return rets

    def ret(self):
        """ Unconditional return """
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
        """ Conditional stock prices in each state - outputs (n_states x 1) array """
        A = self.beta * self.x**(1-self.gamma) * self.Pi
        b = A.sum(axis=1)

        I = np.identity(2)
        S = np.linalg.inv(I - A) @ b

        self.prices_ = S
        return S

    def realized_rets(self):
        """ Realized returns when going from state i to j - outputs an (n_states x n_states) array """
        re = np.zeros((2, 2))
        S = self.prices()
        for i in range(2):
            for j in range(2):
                re[i, j] = self.x[j] * (S[j] + 1) / S[i] - 1

        self.realized_rets_ = re
        return re

    def rets(self):
        """ Conditional expected returns in each state - outputs an (n_states x 1) array """
        real_rets = self.realized_rets()  # NxN matrix
        self.rets_ = (real_rets * self.Pi).sum(axis=1)
        return self.rets_

    def ret(self):
        """ Unconditional expected return on stocks """
        self.ret_ = self.pi_bar @ self.rets()
        return self.ret_

    
