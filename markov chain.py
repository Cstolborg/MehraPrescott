# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:31:54 2021

@author: frede
"""

# Import dataset:
import os
import pandas as pd


# import the dataset to excel:
df = pd.read_excel("./" + 'PCE growth data.xlsx', index_col="Year").dropna()
#df = df + 1

import matplotlib.pyplot as plt
date = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")

plt.figure(figsize=(15,5))
plt.plot(date, df.Growth, label ='Consumption growth', color = 'Steelblue')

plt.xlabel('Time')
plt.ylabel('Pct.')
plt.legend(frameon=False)

summary_stat = df.describe()

# Estimate the AR(1) model:
from statsmodels.tsa.ar_model import AutoReg

mod = AutoReg(df.Growth, 1, old_names=False)
res = mod.fit(cov_type="HC0")
print(res.summary())

# Compute the properties of the AR(1) process:
import numpy as np

sigma = np.sqrt(res.sigma2)
delta = res.params[0]
rho = res.params[1]

ar_mean = delta / (1-rho)
ar_var = sigma**2 / (1-rho**2)
ar_std = np.sqrt(ar_var)
ar_rho = rho*ar_var

# Compute transition matrix and state vector in Markov Chain:
import quantecon
from quantecon import markov

m = markov.rouwenhorst(n = 10, ybar = delta, sigma = sigma, rho = rho)
P = m.P
Z = m.state_values + 1

# Compute unconditional probabilities (ergodic distribution):
eig_val, eig_vec = np.linalg.eig(P.T)
eig_vec = eig_vec[np.argmax(eig_val)]
pi_bar = eig_vec / eig_vec.sum()

# Compute unconditional probabilities (ergodic distribution) with brute force:
pi_bar = np.linalg.matrix_power(P, 1000)[0]
#print("pi_bar = brute_force? ", pi_bar == pi_brute)

# Compute the properties of the markov chain:
m_mean = pi_bar @ Z
m_std = np.sqrt(pi_bar @ Z**2 - (pi_bar @ Z)**2)
m_persistence = np.trace(P) - 1

check = pd.DataFrame([[ar_mean, ar_std, rho], [m_mean-1, m_std, m_persistence]], columns = ['Mean', 'Std', 'rho'])
check