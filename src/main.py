import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.mehra_economy import MehraModel
from src.calibrate_mc_from_data import CalibrateMcChainFromData
from src.markov_chain import MarkovChain

plt.rcParams.update({'font.size': 15})

def plot_economy(rf, re, ER, betas, gammas):
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    #plt.tight_layout()
    fig.subplots_adjust(hspace=.4)

    axs[0, 0].scatter(rf, ER)
    axs[0, 1].scatter(betas, ER)
    axs[1, 0].scatter(gammas, rf, color='b', label='Rf')
    axs[1, 0].scatter(gammas, re, color='r', label='Re')
    axs[1, 0].legend()

    axs[1, 1].scatter(gammas, ER)

    for ax, xlab, ylab in zip(axs.flatten(), ['Rf', r'$\beta$', r'$\gamma$', r'$\gamma$'],
                              ['Excess Return', 'Excess Return', 'Return', 'Excess Return']):
        ax.set_xlabel(xlab)
        ax.set_title(ylab)

    plt.show()


if __name__ == '__main__':
    method = "NIPA"
    if method == "NIPA":
        # Read data and calibrate a Markov chain on an AR(1) process on yearly growth
        data = pd.read_excel("../data/PCE growth data.xlsx", index_col="year")
        calibration = CalibrateMcChainFromData(data, n_states=10)
        mc = calibration()  # Call method returns a MarkovChain
    elif method == "MehraMarkovChain":
        p = 0.43
        x = np.array([ 1 +0.018 - 0.036, 1+ 0.018 + 0.036])
        mc = MarkovChain(Pi=np.array([[p, 1 - p], [1 - p, p]]), x=x)

    # Initialize economy object from Markov Chain
    model = MehraModel(mc)
    model.calibrate_beta_to_rf(target_rf=0.05)

    print("Equity prices = ", model.stocks.prices_)
    print("Equity realized returns: \n", model.stocks.realized_rets_)
    print("Equity conditional returns: \n", model.stocks.rets_)
    print("Equity unconditional returns: \n", model.stocks.ret_)

    print('-' * 50)
    print("Excess return: ", model.excess_ret())


    # Plot admissible region of equity premiums for different pairs (beta, gamma)
    N = 100
    ER, rf, re, betas, gammas = model.num_experiment(N=N, rf_bound=(0.01, 0.04))
    plot_economy(rf, re, ER, betas, gammas)

    # Repeat procedure but less restriction on bounds of the risk-free rate
    ER, rf, re, betas, gammas = model.num_experiment(N=N, rf_bound=(0.01, 0.22))
    plot_economy(rf, re, ER, betas, gammas)