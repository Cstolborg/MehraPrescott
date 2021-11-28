import numpy as np

class MarkovChain:
    """
    Class for a finite Markov Chain. Stores information
    such as transition matrix (Pi), stationary distribution (pi_bar) and state values (x).
    """

    def __init__(self, Pi, x):
        self.Pi = Pi
        self.x = x
        self.n_states = len(x)

        self.pi_bar = self.stationary_dist()
        self.mu, self.rho, self.std = self.uncond_moements()

    def stationary_dist(self):
        # Compute unconditional probabilities (ergodic distribution):
        eig_val, eig_vec = np.linalg.eig(self.Pi.T)
        eig_vec = eig_vec[:, np.argmax(eig_val)]
        pi_bar = eig_vec / eig_vec.sum()
        return pi_bar

    def uncond_moements(self):
        if not hasattr(self, 'pi_bar'):  # Check pi_bar is computed
            self.stationary_dist()

        mean = self.pi_bar @ self.x
        std = np.sqrt(self.pi_bar @ self.x ** 2 - (self.pi_bar @ self.x) ** 2)
        rho = np.trace(self.Pi) - 1

        return mean, rho, std

    def __repr__(self):
        msg = "Markov chain with transition matrix \nPi = \n{0}"
        msg = msg + "\nand stationary distribution \n{1}"
        return msg.format(self.Pi, self.pi_bar)

    def __str__(self):
        return str(self.__repr__)

if __name__ == '__main__':
    p = 0.73975015
    Pi = np.array([[p, 1-p],
                   [1-p, p]])
    x = np.array([0.99657272, 1.03919074])
    mc = MarkovChain(Pi, x)
