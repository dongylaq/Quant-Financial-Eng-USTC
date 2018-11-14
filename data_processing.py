# Created by Yulong Dong
#      -- current PhD student at UC Berkeley, Berkley, CA, United States
#      -- dongyl at berkeley.edu
# Description:
#      -- This code is for course project of 204X04 Quantitative Financial Engineering: From Theory to Practice at University of Sciece and Technology of China, USTC
#      -- two args are needed to use this code, data_file_path (csv), result_saving_path
#      -- python data_processing data_file saving_path
#      -- data can be gathered by another python code


import numpy as np
import pandas as pd
import scipy.optimize as sco
import scipy.interpolate as intpl
import matplotlib.pyplot as plt
import tushare as ts

import sys
from subprocess import *

argv = sys.argv
if len(argv) != 3 :
    print 'Error Input'
    sys.exit()

data_file_name = argv[1]
save_path = argv[2]

out_file = open("output.txt", "w")

data = pd.read_excel(data_file_name)

data.index = data['date'].tolist()
data.pop('date')
# number of assets
N_of_Assets = data.columns.size

(data/data.ix[0]).plot(figsize=(10,8), grid=True)
plt.savefig(save_path+'/Fig1')
log_returns = np.log(data / data.shift(1))
log_returns.hist(bins=50, figsize=(12, 9))
plt.savefig(save_path+'/Fig2')
# use log return to be the return
rets = log_returns
# calculate return for year
year_ret = rets.mean() * 252
# calculate covariance matrix
year_volatility = rets.cov() * 252
print >> out_file, '========== Info of {0} Assets =========='.format(N_of_Assets)
print >> out_file, '---------- mean ----------'
print >> out_file, year_ret
print >> out_file, '---------- covariance matrix ----------'
print >> out_file, year_volatility
print >> out_file, '---------- correlation coefficient matrix ----------'
print >> out_file, rets.corr()

def rand_vect(number_of_assets):
    weights = np.random.random(number_of_assets)
    weights /= np.sum(weights)
    return weights
# function to calculate return, volatility, Sharpe ratio
def data_statistic(weights):
    statistic_ret = np.sum(rets.mean() * weights) * 252
    statistic_vol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([statistic_ret, statistic_vol, statistic_ret/statistic_vol])

# assume we have 10^4 portfolios randomly, use Monte Carlo to do estimation
port_ret = []
port_vol = []
for i in range(10000):
    weights = rand_vect(N_of_Assets)
    port_ret.append(np.sum(rets.mean() * weights) * 252)
    port_vol.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))

port_ret = np.array(port_ret)
port_vol = np.array(port_vol)

# use the optimization in scipy to minimize the ( - Sharpe ratio )
def minus_sharpe_ratio(weights):
    return -data_statistic(weights)[2]
bnds = tuple((0, 1) for x in range(N_of_Assets))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
opts = sco.minimize(minus_sharpe_ratio, N_of_Assets * [1. / N_of_Assets, ], method = 'SLSQP', bounds = bnds, constraints = cons)
print >> out_file, '========== optimal portfolio wrt Sharpe Ratio =========='
print >> out_file, '\t'.join(map(str, opts['x'].tolist()))
print >> out_file, '---------- optimal return, volatity, Sharpe ratio ----------'
print >> out_file, '\t'.join(map(str, data_statistic(opts['x']).tolist()))

def portfolio_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
# use the minimizing variance to calculate optimal portfolio
def min_variance(weights):
    return portfolio_vol(weights) ** 2
optv = sco.minimize(min_variance, N_of_Assets * [1. /N_of_Assets, ], method = 'SLSQP', bounds = bnds, constraints = cons)
print >> out_file, '========== optimal portfolio wrt Minimizing Variance =========='
print >> out_file, '\t'.join(map(str, optv['x'].tolist()))
print >> out_file, '---------- optimal return, volatity, Sharpe ratio ----------'
print >> out_file, '\t'.join(map(str, data_statistic(optv['x']).tolist()))

# calculate the efficient boundary
min_bnd = np.min(port_ret)-0.02
max_bnd = np.max(port_ret)+0.02
target_ret = np.linspace(min_bnd, max_bnd, 30)
def cal_target_vol(ret):
    cons = ({'type': 'eq', 'fun': lambda x: data_statistic(x)[0]-ret}, {'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    res = sco.minimize(portfolio_vol, N_of_Assets*[1. /N_of_Assets, ], method = 'SLSQP', bounds = bnds, constraints = cons)
    return res['fun']
target_vol = [cal_target_vol(target_ret[i]) for i in range(30)]
# smooth fit of efficient boundary
#fp2 = np.polyfit(target_ret, target_vol, 6)
#f2 = np.poly1d(fp2)
fp2 = intpl.splrep(target_ret, target_vol, s=0)
fx = np.linspace(min_bnd, max_bnd, 1000)

plt.figure(figsize=(9,5))
plt.scatter(port_vol, port_ret, c=port_ret/port_vol, marker = 'o')
# efficient boundary
plt.scatter(target_vol, target_ret, c=target_ret/target_vol, marker='x')
eff_bnd, = plt.plot(intpl.splev(fx, fp2, der=0), fx, linewidth=1.5, color='b')
plt.legend([eff_bnd], ['efficient boundary'], loc='upper left', fontsize='small')
# red star - Sharpe ratio, yellow star - minimizing variance
plt.plot(data_statistic(opts['x'])[1], data_statistic(opts['x'])[0], 'r*', markersize = 15.0)
plt.annotate(r'Max Sharpe', xy=(data_statistic(opts['x'])[1], data_statistic(opts['x'])[0]), xycoords='data', xytext=(+10, +30), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.plot(data_statistic(optv['x'])[1], data_statistic(optv['x'])[0], 'y*', markersize = 15.0)
plt.annotate(r'Min Variance', xy=(data_statistic(optv['x'])[1], data_statistic(optv['x'])[0]), xycoords='data', xytext=(+10, +30), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe Ratio')
plt.savefig(save_path+'/Fig3')
# calculate the market line
# find argmin target_volatility and upper_half
ind_min_targ_vol = np.argmin(target_vol)
upper_half_vol = target_vol[ind_min_targ_vol:]
upper_half_ret = target_ret[ind_min_targ_vol:]
# fit upper half vol
tck = intpl.splrep(upper_half_vol, upper_half_ret)
def up_half_f(x):
    return intpl.splev(x, tck, der=0)
def up_half_d_f(x):
    return intpl.splev(x, tck, der=1)
# find point
def find_pt(p, risk_free_ret = 0.02):
    eq1 = risk_free_ret - p[0]
    eq2 = risk_free_ret + p[1] * p[2] - up_half_f(p[2])
    eq3 = p[1] - up_half_d_f(p[2])
    return eq1, eq2, eq3
opt_pt = sco.fsolve(find_pt, [0.01, 0.50, 0.15])
print >> out_file, '========== market line wrt tangent point =========='
print >> out_file, '\t'.join(map(str, opt_pt.tolist()))

# market line
plt.figure(figsize=(9,5))
plt.scatter(port_vol, port_ret, c=port_ret/port_vol, marker = 'o')
# efficient boundary
plt.scatter(target_vol, target_ret, c=target_ret/target_vol, marker='x')
eff_bnd, = plt.plot(intpl.splev(fx, fp2, der=0), fx, linewidth=1.5, color='b')
plt.legend([eff_bnd], ['efficient boundary'], loc='upper left', fontsize='small')
# red star - Sharpe ratio, yellow star - minimizing variance
plt.plot(data_statistic(opts['x'])[1], data_statistic(opts['x'])[0], 'r*', markersize = 15.0)
plt.annotate(r'Max Sharpe', xy=(data_statistic(opts['x'])[1], data_statistic(opts['x'])[0]), xycoords='data', xytext=(+10, +30), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.plot(data_statistic(optv['x'])[1], data_statistic(optv['x'])[0], 'y*', markersize = 15.0)
plt.annotate(r'Min Variance', xy=(data_statistic(optv['x'])[1], data_statistic(optv['x'])[0]), xycoords='data', xytext=(+10, +30), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
fx = np.linspace(0, max_bnd)
mark_line, = plt.plot(fx, opt_pt[0]+opt_pt[1]*fx, linewidth=1.5, color='C1')
plt.legend([eff_bnd, mark_line], ['efficient boundary', 'market line'], loc='upper left', fontsize='small')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe Ratio')
plt.savefig(save_path+'/Fig4')
plt.close()
