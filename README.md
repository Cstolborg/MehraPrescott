
```
pip install -r requirements.txt
```



```python
import os

import numpy as np
import pandas as pd
from math import erfc, sqrt
from scipy import stats
import statsmodels.api as sm
import scipy.optimize as opt
import matplotlib.pyplot as plt
import warnings

from utils import rouwenhorst
from src.mehra_economy import MehraEconomy
from src.calibrate_mc_from_data import CalibrateMcChainFromData
from src.markov_chain import MarkovChain

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
```

## 2. Summary statistics of monthly, annual and quarterly growth


```python
df_month = pd.read_csv("./data/PCE monthly growth data.csv", index_col='date')  # Monthly data
df_quart = pd.read_csv("./data/PCE quarterly growth data.csv", index_col='date')
df_annual = pd.read_excel("./data/PCE growth data.xlsx", index_col='year')

def describe(df, stats):
    d = df.describe()
    return d.append(df.reindex(d.columns, axis = 1).agg(stats))

out = pd.DataFrame()
out1 = {}

for df, name in zip([df_month, df_quart, df_annual], ['Monthly', 'Quarterly', 'Annual']):
    out[name] = describe(df, ['skew', 'kurt'])

    out1[name] = []
    for i in [1,2,3]:
        out1[name].append(df['growth'].autocorr(lag=i))

out1 = pd.DataFrame(out1, index=['AR(1)', 'AR(2)', 'AR(3)'])

out.append(out1).round(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Monthly</th>
      <th>Quarterly</th>
      <th>Annual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>743.0000</td>
      <td>295.0000</td>
      <td>91.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.0015</td>
      <td>0.0045</td>
      <td>0.0173</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0071</td>
      <td>0.0094</td>
      <td>0.0220</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.1331</td>
      <td>-0.1109</td>
      <td>-0.0806</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.0007</td>
      <td>0.0020</td>
      <td>0.0099</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0016</td>
      <td>0.0047</td>
      <td>0.0203</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.0038</td>
      <td>0.0075</td>
      <td>0.0305</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.0624</td>
      <td>0.0751</td>
      <td>0.0737</td>
    </tr>
    <tr>
      <th>skew</th>
      <td>-8.4756</td>
      <td>-5.0063</td>
      <td>-1.5460</td>
    </tr>
    <tr>
      <th>kurt</th>
      <td>186.7268</td>
      <td>88.6626</td>
      <td>5.1247</td>
    </tr>
    <tr>
      <th>AR(1)</th>
      <td>0.0352</td>
      <td>-0.1366</td>
      <td>0.4723</td>
    </tr>
    <tr>
      <th>AR(2)</th>
      <td>-0.2615</td>
      <td>0.0098</td>
      <td>0.1854</td>
    </tr>
    <tr>
      <th>AR(3)</th>
      <td>-0.0196</td>
      <td>0.0877</td>
      <td>-0.0530</td>
    </tr>
  </tbody>
</table>
</div>



Lower frequency data (annual) is more persistent than higher frequency data as can be seen by their AR(1) components. Since in the Mehra-Prescott model, $p=\frac{1+\rho}{2}$, then it follows that for $\rho$ close to zero, $p$ will be very close to $0.5$. Thus, for annual data, we can infer that $p>>0.5$ and thus persistence of a Markov Chain will be higher. As a result, all else equal, higher persistence decreases the equity premium in the Mehra-Prescott model. Thus, using the annual data, we expect to find a smaller premium than we would have for monthly data.


```python
path_mehra = "./data/Shiller data extended.xlsx"
out, summ = run(n=2, path=path_mehra)

summ
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/p8/5fpcvc6s2c15g9s6dcp0fqbh0000gn/T/ipykernel_25700/4249535305.py in <module>
          1 path_mehra = "./data/Shiller data extended.xlsx"
    ----> 2 out, summ = run(n=2, path=path_mehra)
          3 
          4 summ


    NameError: name 'run' is not defined


### 10-state Markov Chain


```python
out, summ = run(n=10, path=path_mehra)

summ
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/p8/5fpcvc6s2c15g9s6dcp0fqbh0000gn/T/ipykernel_25700/880964018.py in <module>
    ----> 1 out, summ = run(n=10, path=path_mehra)
          2 
          3 summ


    NameError: name 'run' is not defined


The 10-state chain appear to match the autocorrelation worse than the two-state, which is exact.


```python
path_nipa = "./data/PCE growth data.xlsx"
out, summ = run(n=2, path=path_nipa)

summ
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean</th>
      <th>Rho</th>
      <th>Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AR coef</th>
      <td>0.009307</td>
      <td>0.4795</td>
      <td>0.018700</td>
    </tr>
    <tr>
      <th>AR moments</th>
      <td>1.017882</td>
      <td>0.4795</td>
      <td>0.021309</td>
    </tr>
    <tr>
      <th>Markov Chain</th>
      <td>1.017882</td>
      <td>0.4795</td>
      <td>0.021309</td>
    </tr>
  </tbody>
</table>
</div>



### 10-state chain


```python
_, summ = run(n=10, path=path_nipa)

summ
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean</th>
      <th>Rho</th>
      <th>Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AR coef</th>
      <td>0.009307</td>
      <td>0.479500</td>
      <td>0.018700</td>
    </tr>
    <tr>
      <th>AR moments</th>
      <td>1.017882</td>
      <td>0.479500</td>
      <td>0.021309</td>
    </tr>
    <tr>
      <th>Markov Chain</th>
      <td>1.017882</td>
      <td>0.919996</td>
      <td>0.021309</td>
    </tr>
  </tbody>
</table>
</div>




```python
_, summ = run(n=2, path=path_nipa, start_year=1950)
summ
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean</th>
      <th>Rho</th>
      <th>Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AR coef</th>
      <td>0.008654</td>
      <td>0.493172</td>
      <td>0.013500</td>
    </tr>
    <tr>
      <th>AR moments</th>
      <td>1.017075</td>
      <td>0.493172</td>
      <td>0.015519</td>
    </tr>
    <tr>
      <th>Markov Chain</th>
      <td>1.017075</td>
      <td>0.493172</td>
      <td>0.015519</td>
    </tr>
  </tbody>
</table>
</div>



### 10-state chain


```python
_, summ = run(n=10, path=path_nipa, start_year=1950)
summ
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean</th>
      <th>Rho</th>
      <th>Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AR coef</th>
      <td>0.008654</td>
      <td>0.493172</td>
      <td>0.013500</td>
    </tr>
    <tr>
      <th>AR moments</th>
      <td>1.017075</td>
      <td>0.493172</td>
      <td>0.015519</td>
    </tr>
    <tr>
      <th>Markov Chain</th>
      <td>1.017075</td>
      <td>0.971377</td>
      <td>0.015519</td>
    </tr>
  </tbody>
</table>
</div>



## Calibrating a Markov Chain on annual data


```python
# Calibrate a Markov chain on an AR(1) process on yearly growth
calibration = CalibrateMcChainFromData(df_annual)
mc = calibration(summary=True)  # Call method returns calibrated MarkovChain
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 growth   R-squared:                       0.223
    Model:                            OLS   Adj. R-squared:                  0.214
    Method:                 Least Squares   F-statistic:                     25.27
    Date:                Mon, 29 Nov 2021   Prob (F-statistic):           2.60e-06
    Time:                        14:44:15   Log-Likelihood:                 231.44
    No. Observations:                  90   AIC:                            -458.9
    Df Residuals:                      88   BIC:                            -453.9
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.0093      0.003      3.555      0.001       0.004       0.015
    growth         0.4795      0.095      5.027      0.000       0.290       0.669
    ==============================================================================
    Omnibus:                       32.934   Durbin-Watson:                   1.843
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              125.252
    Skew:                          -1.069   Prob(JB):                     6.34e-28
    Kurtosis:                       8.370   Cond. No.                         48.4
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
mc  # Resulting Markov Chain
```




    Markov chain with transition matrix 
    Pi = 
    [[0.73975015 0.26024985]
     [0.26024985 0.73975015]]
     and state values = [0.99657272 1.03919074]
    and stationary distribution = [0.5 0.5]



The unconditional mean of excess returns is practically zero. These findings do not confirm the empirically observed excess returns. It is puzzling that both risk-free returns and equity returns are both high at the same time. In the example, the risk-free rate is at 5\%, so in the data we should expect risk-free rates to be much higher. Thus, it is puzzling risk-free rates are so low empirically.

## Characterizing the Mehra-Prescott Economy from the Markov Chain


```python
# Initialize economy object
beta = np.exp(-0.02)
gamma = 2.
econ = MehraEconomy(mc, beta=beta, gamma=gamma)
econ(beta, gamma)  # Run call method to generate asset prices and returns

econ.excess_ret_
```




    0.00010089558890696848




```python
[(key, econ.bonds.__dict__[key]) for key in econ.bonds.__dict__ if key in ['prices_', 'rets_', 'ret_']]
```




    [('prices_', array([0.96631663, 0.92829637])),
     ('rets_', array([0.03485749, 0.07724217])),
     ('ret_', 0.056049830814633106)]




```python
[(key, econ.stocks.__dict__[key]) for key in econ.stocks.__dict__ if key in ['prices_', 'rets_', 'ret_']]
```




    [('prices_', array([27.09263073, 26.10059777])),
     ('rets_', array([0.03495448, 0.07734697])),
     ('ret_', 0.056150726403540074)]



Stock and bond returns are quite similar for these parameter values. Next we check if this is a general property of the model for many parameter combinations.

## Numerical experiment

In this section, we check the admissible region of excess returns for different pairs of ($\beta$, $\gamma$). Empirically, excess returns should be around 6\%, so we should expect the model to deliver such a return.


```python
plt.rcParams.update({'font.size': 15})

def plot_economy(rf, re, ER, betas, gammas):
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    #plt.tight_layout()
    fig.subplots_adjust(hspace=.4)

    axs[0, 0].scatter(rf, ER)
    axs[0, 1].scatter(betas, ER)
    axs[1, 0].scatter(gammas, rf, color='b', label='Risk-free')
    axs[1, 0].scatter(gammas, re, color='r', label='Stocks')
    axs[1, 0].legend()

    axs[1, 1].scatter(gammas, ER)

    for ax, xlab, ylab in zip(axs.flatten(), ['Rf', r'$\beta$', r'$\gamma$', r'$\gamma$'],
                              ['Excess Return', 'Excess Return', 'Return', 'Excess Return']):
        ax.set_xlabel(xlab)
        ax.set_title(ylab)

    plt.show()
```

### Admissible region of excess returns for differing paris ($\beta$, $\gamma$)

With the constraint that risk-free returns $\in (0, 0.04)$

The data is not able to reconstruct the empirical data. No combination of $\beta$ and $\gamma$ is able to match observed excess returns. Finally, the results here are quite different from those in the Mehra-Prescott paper, but this is due to them having a negative annual autocorrelation, whereas ours is positive. For a replication of their results see the final section. 


```python
N = 100
ER, rf, re, betas, gammas = econ.num_experiment(N=N)
plot_economy(rf, re, ER, betas, gammas)
```


    
![png](main_files/main_26_0.png)
    


### Repeating above experiment, without the constraint that Rf<0.04

From the plot, it is evident that equity returns are increasing in $\gamma$, however so is the risk-free rate, which is even increasing faster than equity returns. Thus resulting in a negative excess return.


```python
ER, rf, re, betas, gammas = econ.num_experiment(N=N, rf_bound=(0.01, 0.22))
plot_economy(rf, re, ER, betas, gammas)
```


    
![png](main_files/main_28_0.png)
    


## Appendix: Replicating Mehra-Prescotts Figure 4

To compare our results with the original ones, we here implement the model using the exact same numbers as used in the Mehra-Prescott paper. 

Upper left corner corresponds to their original figure 4. Interestingly, with their parameters the excess returns are now increasing in $\gamma$ and not decreasing. This is because, as mentioned, now the autocorrelation is negative. Yet, the model is still unable to capture empirical excess returns.


```python
p = 0.43
x = np.array([ 1 +0.018 - 0.036, 1+ 0.018 + 0.036])
mc = MarkovChain(Pi=np.array([[p, 1 - p], [1 - p, p]]), x=x)

# Initialize economy object
econ = MehraEconomy(mc)
econ.calibrate_beta_to_rf(target_rf=0.05)

# Plot
ER, rf, re, betas, gammas = econ.num_experiment(N=N)
plot_economy(rf, re, ER, betas, gammas)
```


    
![png](main_files/main_30_0.png)
    



```python

```
