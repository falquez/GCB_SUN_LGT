#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:09:49 2019

@author: carlos
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from collections import defaultdict
from lmfit import models, Model

from matplotlib import rc
rc('font', size=12)

def groupBy(data, labels):
    grouped = defaultdict(lambda: np.array([],data.dtype))
    for row in data:
        key = ()
        for l in labels:
          key = key + ("{0:.8f}".format(row[l]),)
        grouped[key] = np.concatenate(( np.copy([row]) ,grouped[key]))
    return grouped

def combineData(data1, data2, fs, c_idx):
  result = np.array([],data1.dtype)
  for idx, r1 in enumerate(data1):
    ri = np.zeros((1,), dtype=r1.dtype)
    for name in r1.dtype.names:
      ri[0][name] = r1[name]
    ri[0][c_idx] = fs(data1[idx][c_idx], data2[idx][c_idx])
    result = np.concatenate((ri, result))
  return np.sort(result)

def reduceData(data, fs, r_idx, c_idx):
  data_desc = [t for t in data.dtype.descr if t[0] not in [c_idx]]
  dt = np.dtype( data_desc )
  result = np.array([],dt)
  #initialize reduced row
  def initrrow(row):
    rrow = np.zeros((1,), dtype=dt)
    for name in set(data.dtype.names).intersection(set(rrow.dtype.names)):
      rrow[name] = row[name]
    rrow[r_idx]=fs(row[r_idx], row[c_idx])
    return rrow
  ri = initrrow(data[0])
  for row in data[1:]:
    if(row[c_idx]!=0):
      ri[r_idx] += fs(row[r_idx], row[c_idx])
    else:
      #concatenate
      result = np.concatenate((ri, result))
      ri = initrrow(row)
  return np.sort(np.concatenate(( ri , result)))

def reduceFitData(data, model, x, y, param_dict):
    min_len = np.size(model.param_names)
    data_desc = [t for t in data[list(data.keys())[0]].dtype.descr if t[0] not in [x,y]]
    dt = np.dtype( data_desc + list(map(lambda x: (x,'<f8'), model.param_names)) + [('d'+y,'<f8')] )

    result = np.array([], dt )

    for k,v in data.items():
        if(len(v)>=min_len):
            params = {'params': model.make_params(**param_dict), x: v[x]}
            fit = model.fit(v[y], **params)
            #print(str(k)+":")
            #print(fit.fit_report())
            
            ri = np.zeros((1,), dtype=result.dtype)
            for desc in data_desc:
                ri[0][desc[0]] = v[0][desc[0]]
            for name in model.param_names:
                ri[0][name] = fit.best_values[name]
            ri[0]['d'+y]=fit.params[y].stderr
            result = np.concatenate((ri, result))
    return np.sort(result)

currDir = os.path.dirname(os.path.realpath(__file__))
dataDir = currDir + "/"

entropies = np.sort(np.loadtxt( dataDir+"entropies.txt", ndmin=1, dtype=np.dtype([('D', '<i4'), ('L', '<i4'), ('g', '<f8'), ('r', '<f8'), ('x', '<f8'), ('y', '<f8'), ('R', '<i4'), ('i', '<i4'), ('S', '<f8')])), order=['L', 'g', 'r','D', 'R', 'i'])

# p_j = Tr(\rho_j) = Tr(\Sigma^2)
p_j = reduceData(entropies, lambda p, i: p*p, 'S', 'i')

# S_j = S(\rho_j) = -Tr(\Sigma^2 log(\Sigma^2))
S_j = reduceData(entropies, lambda p, i: -p*p*np.log2(p*p), 'S', 'i')

# S_dist_j = p_j*S_j = -Tr(\Sigma^2 log(\Sigma^2 / Tr (\Sigma^2)))
#                    = -Tr(\Sigma^2 log(\Sigma^2)) + Tr(\Sigma^2) log(Tr(\Sigma^2)) 
#                    = S_j + p_j*log(p_j)
S_dist_j = combineData(p_j, S_j, lambda p, s: s+p*np.log2(p), 'S')

# S_dist = \sum_j S_dist_j
S_dist = reduceData(S_dist_j, lambda s, r: s, 'S', 'R')

# S_class = -\sum_j p_j log(p_j)
S_class = reduceData(p_j, lambda p, r: -p*np.log2(p), 'S', 'R')

# S_rep = \sum_j p_j log(2j+1)
S_rep = reduceData(p_j, lambda p, r : p*np.log2(r+1), 'S', 'R',)

S_all = reduceData(S_j, lambda s, r: s, 'S', 'R')

modelD = models.ExpressionModel("S + F1/D", independent_vars=['D'])
modelLog = Model(lambda g, c,b,s: -(c/6)*np.log2(g) + b*g + s )


reducedS_dist  = reduceFitData(groupBy(S_dist,  ['L', 'g', 'r']), modelD, 'D', 'S', {'S':-2,'F1':1})
reducedS_class = reduceFitData(groupBy(S_class, ['L', 'g', 'r']), modelD, 'D', 'S', {'S':-2,'F1':1})

reducedS_all = reduceFitData(groupBy(S_all, ['L', 'g', 'r']), modelD, 'D', 'S', {'S':-2,'F1':-0.1})
reducedS_rep = reduceFitData(groupBy(S_rep, ['L', 'g', 'r']), modelD, 'D', 'S', {'S':-2,'F1':-0.1})

reducedS=np.array([],reducedS_all.dtype)
for idx, r in enumerate(reducedS_all):
    ri = np.zeros((1,), dtype=r.dtype)
    for name in r.dtype.names:
        ri[0][name] = r[name]
    ri[0]['S'] = r['S'] + reducedS_rep[idx]['S']
    ri[0]['dS'] = np.sqrt(r['dS']*r['dS'] + reducedS_rep[idx]['dS']*reducedS_rep[idx]['dS'])
    reducedS = np.concatenate((ri, reducedS))

data100_04 = reducedS[np.where((reducedS['L']==100) &  np.isclose(reducedS['r'],0.4)) ]
data100_08 = reducedS[np.where((reducedS['L']==100) &  np.isclose(reducedS['r'],0.8)) ]
data100_12 = reducedS[np.where((reducedS['L']==100) &  np.isclose(reducedS['r'],1.2)) ]

data200_04 = reducedS[np.where((reducedS['L']==200) &  np.isclose(reducedS['r'],0.4)) ]
data200_08 = reducedS[np.where((reducedS['L']==200) &  np.isclose(reducedS['r'],0.8)) ]
data200_12 = reducedS[np.where((reducedS['L']==200) &  np.isclose(reducedS['r'],1.2)) ]

fit100_04 = modelLog.fit(data100_04['S'],g=data100_04['g'], c=2, b=0,s=1)
fit100_08 = modelLog.fit(data100_08['S'],g=data100_08['g'], c=2, b=0,s=1)
fit100_12 = modelLog.fit(data100_12['S'],g=data100_12['g'], c=2, b=0,s=1)

fit200_04 = modelLog.fit(data200_04['S'],g=data200_04['g'], c=2, b=0,s=1)
fit200_08 = modelLog.fit(data200_08['S'],g=data200_08['g'], c=2, b=0,s=1)
fit200_12 = modelLog.fit(data200_12['S'],g=data200_12['g'], c=2, b=0,s=1)

csvS = np.zeros((6,), np.dtype([ ('L', '<i4'), ('r', '<f8'), ('c', '<f8'), ('dc', '<f8'), ('de', '<f8')]) )

csvS[0]['L']  = data100_04[0]['L']
csvS[0]['r']  = data100_04[0]['r']
csvS[0]['c']  = fit100_04.params['c'].value
csvS[0]['dc'] = fit100_04.params['c'].stderr
csvS[0]['de'] = 2.0 - fit100_04.params['c'].value

csvS[1]['L']  = data200_04[0]['L']
csvS[1]['r']  = data200_04[0]['r']
csvS[1]['c']  = fit200_04.params['c'].value
csvS[1]['dc'] = fit200_04.params['c'].stderr
csvS[1]['de'] = 2.0 - fit200_04.params['c'].value

csvS[2]['L']  = data100_08[0]['L']
csvS[2]['r']  = data100_08[0]['r']
csvS[2]['c']  = fit100_08.params['c'].value
csvS[2]['dc'] = fit100_08.params['c'].stderr
csvS[2]['de'] = 2.0 - fit100_08.params['c'].value

csvS[3]['L']  = data200_08[0]['L']
csvS[3]['r']  = data200_08[0]['r']
csvS[3]['c']  = fit200_08.params['c'].value
csvS[3]['dc'] = fit200_08.params['c'].stderr
csvS[3]['de'] = 2.0 - fit200_08.params['c'].value

csvS[4]['L']  = data100_12[0]['L']
csvS[4]['r']  = data100_12[0]['r']
csvS[4]['c']  = fit100_12.params['c'].value
csvS[4]['dc'] = fit100_12.params['c'].stderr
csvS[4]['de'] = 2.0 - fit100_12.params['c'].value

csvS[5]['L']  = data200_12[0]['L']
csvS[5]['r']  = data200_12[0]['r']
csvS[5]['c']  = fit200_12.params['c'].value
csvS[5]['dc'] = fit200_12.params['c'].stderr
csvS[5]['de'] = 2.0 - fit200_12.params['c'].value

np.savetxt(dataDir+"entropy.csv", csvS, header="m/g L c dc de")

t = np.arange(0.07, 0.16, 0.001)
f1 = -(fit100_08.best_values['c']/6)*np.log2(t) + fit100_08.best_values['b']*t + fit100_08.best_values['s']
f2 = -(fit200_08.best_values['c']/6)*np.log2(t) + fit200_08.best_values['b']*t + fit200_08.best_values['s']

fig, ax = plt.subplots()

ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.002))
ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.01))
ax.grid(True)
ax.set(xlabel=r"$\hat{g}$", ylabel=r"$S$")

ax.scatter(x=data200_08['g'], y=data200_08['S'], marker='o', c='c')
ax.plot(t, f2)

plt.savefig(dataDir+'img/entropy_200_08.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)