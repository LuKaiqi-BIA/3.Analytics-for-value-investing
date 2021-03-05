#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:06:22 2021

@author: kevinlu
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as scs
from scipy.stats.mstats import winsorize

#%%
#a) Collect annual financial statement data on a sample of U.S. companies from 
#Compustat for the period Dec 2003 to Dec 2019 (16 years).
df = pd.read_csv('uscompustat.csv')
df = df.sort_values(by=['gvkey','fyear'])
df = df[df.gsector != 40]     # Delete banks and financial institutions

#%%
#a) Calculate accounting metrics

#    op_prof: operating profitability
#    op_margin: operating margin
#    noa: net operating assets = at - che - (lct - dlc)
#    aturn: total asset turnover
#    

df["op_prof"] = df.groupby(["gvkey"])["ebit"].pct_change()
df['op_margin'] = df["ebit"].shift(1) / df["revt"].shift(1)
# shift(1) because the op_margin is calcualted by (prior year ebit)/(prior year rev)

df["noa"] = df["at"] - df["che"] - (df["lct"] - df["dlc"])
df['aturn'] = df["revt"].shift(1) / df['noa'].shift(1)
# shift(1) because the op_margin is calcualted by (prior year rev)/(prior year noa)
#%%

# Keep relevant variables and delete observations with missing or infinite values
#    Compute summary statistics of variables

df = df[['gvkey','fyear','op_prof','op_margin','aturn']]
df = df[np.isfinite(df)]
df = df.dropna()
print('Descriptive statistics of variables (non-winsorized) \n', 
      df[['op_prof','op_margin','aturn']].describe().round(3), end='\n'*5)

# Winsorize outliers of each variable at 2.5% of lower & 2.5% of upper tails of distribution
for var in ['op_prof','op_margin','aturn']:
   df[var] = winsorize(df[var], limits=[0.025,0.025])
print('Descriptive statistics of variables (winsorized at 5% each tail) \n', 
      df[['op_prof','op_margin','aturn']].describe().round(3), end='\n'*5)

#%%
# Compute Pearson correlation coefficients
print('Correlation coefficients of variables \n', 
      df[['op_prof','op_margin','aturn']].corr().round(4), end='\n'*5)

#%%
#bi) Pooled OLS regression of current-year profitability against previous-year 
#operating profit margin and operating asset turn, with heteroskedasticity-consistent 
#t-tests on the regression coefficients


# Set up matrix of x-variables with intercept
X = sm.add_constant(df[['op_margin','aturn']])

# OLS REGRESSION
# Pooled OLS regression of operating profitability on accounting variables with heteroskedasticity-consistent std errors (HC3)
reg1 = sm.OLS(df["op_prof"], X).fit()
reg1_het = reg1.get_robustcov_results(cov_type='HC3')
print('Pooled OLS regression of future operating profitability against accounting variables with HC std errors \n', 
      reg1_het.summary(), end='\n'*5)

#%%
# bii)Year-by-year OLS regression of the above form, with t-tests on the 
#time-series of regression coefficients.

def olsreg(d, yvar, xvars):
    Ygrp = d[yvar]
    Xgrp = sm.add_constant(d[xvars])
    reg = sm.OLS(Ygrp, Xgrp).fit()
    return reg.params

df_group = df.groupby('fyear')
yearcoef = df_group.apply(olsreg, 'op_prof', ['op_margin','aturn'])
print('Coefficients of year-by-year regressions\n', yearcoef, '\n'*3)
tstat, pval = scs.ttest_1samp(yearcoef, 0)
print('T-statistics and p-values of year-by-year coefficients: \n')
print(pd.DataFrame({'t-stat': tstat.round(4), 'p-value': pval.round(4)}, 
                   index=['const', 'op_margin','aturn']), '\n'*5)

#%%
# biii)Pooled logistic regression of current-year profitability indicator variable (𝐷+,!) 
#against previous- year operating profit margin and operating asset turn

df['d_op_prof'] = np.where(df['op_prof'] > 0, 1, 0)
print(df, '\n'*5)
print('Descriptive statistics of variables \n', 
      df[['d_op_prof', 'op_prof','op_margin','aturn']].describe().round(4), '\n'*5)
reg2 = sm.Logit(df['d_op_prof'], X).fit()
print('Logit regression of operating profitability dummy against accounting variables \n', 
      reg2.summary(), end='\n'*5)
