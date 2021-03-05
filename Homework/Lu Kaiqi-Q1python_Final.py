#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:53:38 2021

@author: kevinlu
"""

import pandas_datareader as pdr
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.diagnostic as sd
from scipy.stats import ttest_ind
from scipy.stats import t
from scipy.stats import levene

#%%
#a) Collect S&P500 prices from Yahoo from 1985-12-01 to 2020-12-01
mkt = pdr.get_data_yahoo("^GSPC", "1985-12-01", "2020-12-31", interval="m")

#%%
#b)
#i)
mkt["mktret"] = mkt.sort_values(by="Date")["Adj Close"].pct_change()
mkt = mkt[["mktret"]].dropna().reset_index()


#ii)
print(mkt["mktret"].describe())

#%%
#b)
#(iii) Test the hypotheses that the returns for the month of January are higher 
#or more positive, while those for the month of September are lower or more negative, 
#than those for other months, using the following statistical tests:
#• t-test of difference in mean returns between January and non-January months;

def independent_t_test(data1, data2, alpha):
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = float(data1.mean()), float(data2.mean())
    var1, var2 = float(data1.var(ddof=1)), float(data2.var(ddof=1))
    
    # Calculate degree of freedom
    dof = n1 + n2 - 2
    
    # Calculate pooled sample variance
    pooled_samp_var = ((n1-1)*var1+(n2-1)*var2)/dof
    
    # Calculate t-statistics
    t_stats = (mean1 - mean2)/np.sqrt((pooled_samp_var/n1)+(pooled_samp_var/n2))
    
    # Calculate critical value and p-value
    cv = t.ppf(1.0 - alpha, dof)
    p = (1.0 - t.cdf(abs(t_stats), dof))
    
    return t_stats, dof, cv, p

jan_ret = mkt.loc[mkt['Date'].dt.month == 1]
no_jan_ret = mkt.loc[mkt['Date'].dt.month != 1]

# Perform Levene's test to test population variances
print(levene(jan_ret["mktret"], no_jan_ret["mktret"]))

# As Levene's test give a p-value of 0.361
#Assume that the population variances are unknown but equal
#H0: jan_mean = no_jan_mean
#H1: jan_mean > no_jan_mean
#Perform a independent one tailed t-test 

t_stat1, dof1, cv1, p1 = independent_t_test(jan_ret, no_jan_ret, 0.05)
print("t_stats={}, Degree of freedom={}, Critical Value={}, p-val={}".format(t_stat1, dof1, cv1, p1))

# Check
stat1, p1 = ttest_ind(jan_ret["mktret"], no_jan_ret["mktret"])
print("p-value={}".format(p1/2))


#%%
#• t-test of difference in mean returns between September and non-September months;
sep_ret = mkt.loc[mkt['Date'].dt.month == 9]
no_sep_ret = mkt.loc[mkt['Date'].dt.month != 9]

# Perform Levene's test to test population variances
print(levene(sep_ret["mktret"], no_sep_ret["mktret"]))

# As Levene's test give a p-value of 0.280
# Assume that the population variances are unknown but equal
#H0: sep_mean = no_sep_mean
#H1: sep_mean > no_sep_mean
#Perform a independent one tailed t-test 


t_stat2, dof2, cv2, p2 = independent_t_test(sep_ret, no_sep_ret, 0.05)
print("t_stats={}, Degree of freedom={}, Critical Value={}, p-val={}".format(t_stat2, dof2, cv2, p2))

# Check
stat2, p2 = ttest_ind(sep_ret["mktret"], no_sep_ret["mktret"])
print("p-value={}".format(p2/2))

#%%
#• an OLS regression of monthly S&P500 returns as dependent variable and two dummy independent variables

mkt["djan"] = np.where(mkt["Date"].dt.month == 1, 1, 0)
mkt["dsep"] = np.where(mkt["Date"].dt.month == 9, 1, 0)

# 5. OLS regression of Citigroup returns against S&P500 returns
mktreg = smf.ols(formula='mktret ~ djan + dsep', data=mkt).fit()
print(mktreg.summary(), '\n'*5)

#White test for heteroskedasticity
het_test = sd.het_white(mktreg.resid, mktreg.model.exog)
labels = ['LM-Statistic', 'LM p-value', 'F-Statistic', 'F-Test p-value']
print('White test for heteroskedasticity:\n', pd.DataFrame([labels, het_test]), 
      '\n'*5)


#OLS regression with heteroskedasticity-consistent std errors (HC3)
mktreg_het = smf.ols(formula='mktret ~ djan + dsep',data=mkt).fit(cov_type='HC3')
print('OLS regression of mktret against dummy variables under heteroskedasticity-consistent std errors:\n', 
      mktreg_het.summary(), '\n'*5)



# Define a function to run OLS regression with autocorrelation-consistent std errors 
#    with various lags (HAC)
def mktreg_autocor(nlags):
    mktreg_autocor_consistent = mktreg.get_robustcov_results(cov_type='HAC', maxlags=nlags)
    print('OLS regression of mktret against dummy variables under autocorrelation-consistent std errors:\n',
          mktreg_autocor_consistent.summary(), '\n'*3)

for nlags in [1, 3, 6, 12]:
    mktreg_autocor(nlags)
    

