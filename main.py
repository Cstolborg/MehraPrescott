import numpy as np
import pandas as pd
from math import erfc, sqrt

def rouwenhorst(n, ybar, sigma, rho):
    r"""
    Takes as inputs n, p, q, psi. It will then construct a markov chain
    that estimates an AR(1) process of:
    :math:`y_t = \bar{y} + \rho y_{t-1} + \varepsilon_t`
    where :math:`\varepsilon_t` is i.i.d. normal of mean 0, std dev of sigma
    The Rouwenhorst approximation uses the following recursive defintion
    for approximating a distribution:
    .. math::
        \theta_2 =
        \begin{bmatrix}
        p     &  1 - p \\
        1 - q &  q     \\
        \end{bmatrix}
    .. math::
        \theta_{n+1} =
        p
        \begin{bmatrix}
        \theta_n & 0   \\
        0        & 0   \\
        \end{bmatrix}
        + (1 - p)
        \begin{bmatrix}
        0  & \theta_n  \\
        0  &  0        \\
        \end{bmatrix}
        + q
        \begin{bmatrix}
        0        & 0   \\
        \theta_n & 0   \\
        \end{bmatrix}
        + (1 - q)
        \begin{bmatrix}
        0  &  0        \\
        0  & \theta_n  \\
        \end{bmatrix}
    Parameters
    ----------
    n : int
        The number of points to approximate the distribution
    ybar : float
        The value :math:`\bar{y}` in the process.  Note that the mean of this
        AR(1) process, :math:`y`, is simply :math:`\bar{y}/(1 - \rho)`
    sigma : float
        The value of the standard deviation of the :math:`\varepsilon` process
    rho : float
        By default this will be 0, but if you are approximating an AR(1)
        process then this is the autocorrelation across periods
    Returns
    -------
    mc : MarkovChain
        An instance of the MarkovChain class that stores the transition
        matrix and state values returned by the discretization method
    """

    # Get the standard deviation of y
    y_sd = sqrt(sigma**2 / (1 - rho**2))

    # Given the moments of our process we can find the right values
    # for p, q, psi because there are analytical solutions as shown in
    # Gianluca Violante's notes on computational methods
    p = (1 + rho) / 2
    q = p
    psi = y_sd * np.sqrt(n - 1)

    # Find the states
    ubar = psi
    lbar = -ubar

    bar = np.linspace(lbar, ubar, n)

    def row_build_mat(n, p, q):
        """
        This method uses the values of p and q to build the transition
        matrix for the rouwenhorst method
        """

        if n == 2:
            theta = np.array([[p, 1 - p], [1 - q, q]])

        elif n > 2:
            p1 = np.zeros((n, n))
            p2 = np.zeros((n, n))
            p3 = np.zeros((n, n))
            p4 = np.zeros((n, n))

            new_mat = row_build_mat(n - 1, p, q)

            p1[:n - 1, :n - 1] = p * new_mat
            p2[:n - 1, 1:] = (1 - p) * new_mat
            p3[1:, :-1] = (1 - q) * new_mat
            p4[1:, 1:] = q * new_mat

            theta = p1 + p2 + p3 + p4
            theta[1:n - 1, :] = theta[1:n - 1, :] / 2

        else:
            raise ValueError("The number of states must be positive " +
                             "and greater than or equal to 2")

        return theta

    theta = row_build_mat(n, p, q)

    bar += ybar / (1 - rho)

    return theta, bar




if __name__ == '__main__':
    mu = 0.0183
    sigma = 0.0357
    rho = -0.14

    P, z = rouwenhorst(n=2, ybar=mu, sigma=sigma, rho=rho)


