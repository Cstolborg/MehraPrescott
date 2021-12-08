import numpy as np


def rouwenhorst(n, mu, sigma, rho):
    r"""
    Takes as inputs n, p, q, psi. It will then construct a markov chain
    that estimates an AR(1)

    Uses mu, sigma and rho directly from AR regression.

    Parameters
    ----------
    n : int
        The number of points to approximate the distribution
    mu : float
        The value :math:`\bar{y}` in the process.  Note that the mean of this
        AR(1) process, :math:`y`, is simply :math:`\bar{y}/(1 - \rho)`
    sigma : float
        The value of the standard deviation of the :math:`\varepsilon` process
    rho : float
        By default this will be 0, but if you are approximating an AR(1)
        process then this is the autocorrelation across periods
    """

    # Get the standard deviation of y
    sigmaz = sigma  # sqrt(sigma**2 / (1 - rho**2))

    p = (1 + rho) / 2

    PI = row_build_mat(n, p, p)  # Build transition matrix

    fi = np.sqrt(n - 1) * sigmaz
    Z = np.linspace(-fi, fi, n)
    Z = Z + mu

    return Z, PI


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