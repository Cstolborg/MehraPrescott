# asset_pricing
A repo for the PhD asset pricing course at CBS


```python
import os

import numpy as np
import pandas as pd
from math import erfc, sqrt
from scipy import stats
import statsmodels.api as sm
import scipy.optimize as opt
import warnings

from utils import rouwenhorst
from main import run

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

## 3.a Reproduce Mehra-Prescott

### Two-state Markov Chain


```python
path_mehra = "./data/Shiller data extended.xlsx"
out, summ = run(n=2, path=path_mehra)

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
      <td>0.022761</td>
      <td>-0.100142</td>
      <td>0.039678</td>
    </tr>
    <tr>
      <th>AR moments</th>
      <td>1.020689</td>
      <td>-0.100142</td>
      <td>0.039878</td>
    </tr>
    <tr>
      <th>Markov Chain</th>
      <td>1.020689</td>
      <td>-0.100142</td>
      <td>0.039878</td>
    </tr>
  </tbody>
</table>
</div>



### 10-state Markov Chain


```python
out, summ = run(n=10, path=path_mehra)

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
      <td>0.022761</td>
      <td>-0.100142</td>
      <td>0.039678</td>
    </tr>
    <tr>
      <th>AR moments</th>
      <td>1.020689</td>
      <td>-0.100142</td>
      <td>0.039878</td>
    </tr>
    <tr>
      <th>Markov Chain</th>
      <td>1.020689</td>
      <td>-0.091026</td>
      <td>0.039878</td>
    </tr>
  </tbody>
</table>
</div>



The 10-state chain appear to match the autocorrelation worse than the two-state, which is exact.

## 3.B NIPA Data long sample
### Two-state chain


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



## 3.C NIPA data post-war
### Two-state chain


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



## 4.

Unconditional moments are matched in 3.B


```python
# Calibrate chain from data
out, summ = run(n=2, path=path_nipa)
Z, P, pi, mu, sigma, rho = out['Z'], out['P'], out['stationary dist'], out['mean'], out['std'], out['rho']
gamma = 2
beta = np.exp(-0.02)

print("Conditional moments")
cond_mu = P @ Z
cond_std = np.sqrt((P @ Z**2) - cond_mu**2)


pd.DataFrame({'mu': cond_mu,
             'std': cond_std}, index=['Bad State', 'Good State'])
```

    Conditional moments





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
      <th>mu</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bad State</th>
      <td>1.007664</td>
      <td>0.0187</td>
    </tr>
    <tr>
      <th>Good State</th>
      <td>1.028099</td>
      <td>0.0187</td>
    </tr>
  </tbody>
</table>
</div>




```python
def bond_return(beta, Z=Z, P=P, n=1, gamma=2):
    # Compute risk-free return from Markov Chain
    B_i = np.zeros(2)
    for i in range(2):
        for j in range(2):
            B_i[i] += beta * P[i, j] * Z[j]**(-gamma)
            
    Rf_i = 1/B_i - 1  # Net conditonal risk-free return
    
    B = pi @ Rf_i
    return B

# Solve for the beta that sets bond-return = 0.05 
beta = opt.fsolve(lambda beta: bond_return(beta) - 0.05, x0=np.exp(-0.02))[0]

beta
```




    0.9858463267717398




```python
def B2_cond_price(beta, Z=Z, P=P, n=1, gamma=2):
    b = np.zeros(n)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                b[i] += P[i, j] * Z[j]**(-gamma) * P[j, k] * Z[k] **(-gamma)
    return b * beta**n
                
B2 = B2_cond_price(beta, n=2)
B2
```




    array([0.9354742 , 0.88157146])




```python
def B1_cond_price(beta, Z=Z, P=P, n=1, gamma=2):
    B1 = np.zeros(2)
    for i in range(2):
        for j in range(2):
            B1[i] += beta * P[i, j] * Z[j]**(-gamma)
    return B1

B1 = B1_cond_price(beta)
            
B1
```




    array([0.9718843 , 0.93364498])




```python
h1 = np.zeros(2)
for i in range(2):
    h1[i] = (B1 / B2[i] * P[i, :]).sum() - 1
    
print(f"Bond returns after 1 period: {h1}")
print(f'Bond prices after 1 period: {1/(1+h1)}')

pd.DataFrame({'B1': B1,
             "B2": B2,
             "Holding-return": h1}, index=['Bad State', 'Good State']).T
```

    Bond returns after 1 period: [0.02828332 0.07035764]
    Bond prices after 1 period: [0.97249462 0.93426716]





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
      <th>Bad State</th>
      <th>Good State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>B1</th>
      <td>0.971884</td>
      <td>0.933645</td>
    </tr>
    <tr>
      <th>B2</th>
      <td>0.935474</td>
      <td>0.881571</td>
    </tr>
    <tr>
      <th>Holding-return</th>
      <td>0.028283</td>
      <td>0.070358</td>
    </tr>
  </tbody>
</table>
</div>



## Price-Dividend ratio in each state i



```python
def equity_price():
    A = beta * Z**(1-gamma) * P
    b = A.sum(axis=1)

    I = np.identity(2)
    S = np.linalg.inv(I - A) @ b
    return S

S = equity_price()
S
```




    array([32.18727579, 31.00294884])




```python
P_D = S / Z
P_D
```




    array([32.29797004, 29.83374248])



## Return on equity


```python
def return_equity():
    re = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            re[j, i] = Z[j]*(S[j]+1)/S[i]-1
    return re

re = return_equity()
re
```




    array([[0.02753442, 0.0667867 ],
           [0.0332396 , 0.07270983]])




```python
# Excess returns
rf = 1/B1 - 1  # 1-period risk-free rate

excess_ret = (re.T - rf).T
excess_ret
```




    array([[-0.00139465,  0.03785764],
           [-0.03783134,  0.00163889]])



## Unconditional moments of equity and bond returns


```python
def uncond_re(re):
    cond_re = (P * re.T).sum(axis=1)
    return pi @ cond_re

re_uncond = uncond_re(re)

uncond_rf = pi @ rf

re_uncond
```




    0.05009376398854534




```python
cond_re = (P * re.T).sum(axis=1)
cond_ER = cond_re - rf

ER_mean = uncond_re(re) - pi @ rf
std = np.sqrt( (pi @ cond_ER**2) - ER_mean**2)
rho1 = np.diag(P)

display(pd.DataFrame([ER_mean, std, rho1], index=['Mean', 'Std', 'Rho']).T)
print("_"*40)
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
      <th>Std</th>
      <th>Rho</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000094</td>
      <td>0.000004</td>
      <td>[0.7397501537169732, 0.7397501537169732]</td>
    </tr>
  </tbody>
</table>
</div>


    ________________________________________



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
