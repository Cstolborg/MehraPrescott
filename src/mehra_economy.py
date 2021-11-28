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
        self.x, self.Pi, self.pi_bar = mc.x, mc.Pi, mc.pi_bar
        self.n_states = len(self.x)

        self.beta = beta
        self.gamma = gamma

    def bond_prices(self, beta=None):
        if beta is None: beta = self.beta
        # Compute risk-free return from Markov Chain
        prices = np.zeros(self.n_states)
        for i in range(self.n_states):
            for j in range(self.n_states):
                prices[i] += beta * self.Pi[i, j] * self.x[j]**(-self.gamma)

        self.bond_prices_ = prices
        return prices

    def bond_rets(self):
        prices = self.bond_prices()
        rets = 1/prices - 1  # Net return
        self.bond_rets_ = rets
        return rets

    def bond_ret(self):
        self.bond_ret_ = self.pi_bar @ self.bond_rets()
        return self.bond_ret_

    def calibrate_beta_to_rf(self, target_rf=0.05):
        pi_bar = self.pi_bar
        Pi = self.Pi
        x = self.x
        gamma = self.gamma
        def bond_price_func(beta):
            prices = np.zeros(self.n_states)
            for i in range(self.n_states):
                for j in range(self.n_states):
                    prices[i] += beta * Pi[i, j] * x[j]**(-gamma)

            return prices

        def target_func(beta):
            Rf_i = 1 / bond_price_func(beta) - 1
            Rf = pi_bar @ Rf_i
            return Rf - target_rf

        self.beta = opt.fsolve(target_func, x0=np.exp(-0.02))[0]

    def equity_prices(self):
        A = self.beta * self.x**(1-self.gamma) * self.Pi
        b = A.sum(axis=1)

        I = np.identity(2)
        S = np.linalg.inv(I - A) @ b

        self.equity_prices_ = S
        return S

    def equity_real_rets(self):
        re = np.zeros((2, 2))
        S = self.equity_prices()
        for i in range(2):
            for j in range(2):
                re[i, j] = self.x[j] * (S[j] + 1) / S[i] - 1

        self.equity_real_rets_ = re
        return re

    def equity_rets(self):
        real_rets = self.equity_real_rets()  # NxN matrix
        self.equity_rets_ = (real_rets * self.Pi).sum(axis=1)
        return self.equity_rets_

    def equity_ret(self):
        self.equity_ret_ = self.pi_bar @ self.equity_rets()
        return self.equity_ret_

    def excess_ret(self):
        self.equity_ret_ = self.equity_ret()
        self.bond_ret_ = self.bond_ret()

        self.excess_ret_ = self.equity_ret_ - self.bond_ret_
        return self.excess_ret_

    def num_experiment(self, N=100):
        rfs = np.linspace(0.01, 0.04, num=N)
        gammas1 = np.linspace(0, 10, num=N)

        gammas = []
        betas = []
        re = []
        rf = []
        ER = []
        for r in rfs:
            for gamma in gammas1:
                self.calibrate_beta_to_rf(target_rf=r)
                if self.beta >= 1.:
                    continue
                self.gamma = gamma
                ER.append(self.excess_ret())
                betas.append(self.beta)
                re.append(self.equity_ret_)
                rf.append(self.bond_ret_)
                gammas.append(gamma)

        return ER, rf, re, betas, gammas





if __name__ == '__main__':
    data = pd.read_excel("../data/PCE growth data.xlsx", index_col="year")
    calibration = CalibrateMcChainFromData(data)
    mc = calibration()  # Call method returns a MarkovChain
    #p = 0.43
    #x = np.array([1+0.018 - 0.036, 1+0.018+0.036])
    #mc = MarkovChain(Pi=np.array([[p, 1-p], [1-p, p]]), x=x)

    econ = MehraEconomy(mc)
    econ.calibrate_beta_to_rf(target_rf=0.05)

    print("Equity prices = ",econ.equity_prices())
    print("Equity realized returns: \n",econ.equity_real_rets())
    print("Equity conditional returns: \n",econ.equity_rets())
    print("Equity unconditional returns: \n", econ.equity_ret())

    print('-'*50)
    print("Excess return: ", econ.excess_ret())

    N = 100
    ER, rf, re, betas, gammas = econ.num_experiment(N=N)


    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))

    axs[0, 0].scatter(rf, re)
    axs[0, 1].scatter(rf, ER)
    axs[1, 0].scatter(betas, ER)
    axs[1, 1].scatter(gammas, ER)

    for ax, xlab, ylab in zip(axs.flatten(), ['Rf', 'Rf', 'beta', 'gamma'],
                              ['Re', 'Excess_ret', 'Excess-ret', 'Excess_ret']):
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    plt.show()

    df = pd.DataFrame({'excess_ret': ER,
                       'Rf': rf,
                       'Re': re,
                       'beta':betas,
                       'gamma':gammas}).set_index('gamma')

    df.plot.scatter(x='Rf', y='excess_ret')