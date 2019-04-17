#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:14:27 2019

@author: carlos
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from collections import defaultdict
from lmfit import models

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

def mapData(data, nf, f):
    dt = np.dtype( [t for t in data.dtype.descr] + [(nf,'<f8')] )
    result = np.array([], dt )
    for idx, r1 in enumerate(data):
        ri = np.zeros((1,), dtype=result.dtype)
        for name in r1.dtype.names:
            ri[0][name] = r1[name]
        ri[0][nf] = f(ri[0])
        result = np.concatenate((ri, result))
    return np.sort(result)

def reduceFitDataE(data, model, x, y, param_dict, epsilon=10E-7):
    modelLen = np.size(model.param_names)
    data_desc = [t for t in data[list(data.keys())[0]].dtype.descr if t[0] not in [x,y]]
    dt = np.dtype( data_desc + list(map(lambda x: (x,'<f8'), model.param_names)) + [('d'+y,'<f8'),('d'+y+'fit','<f8')] )

    result = np.array([], dt )

    for k,v in data.items():
        if(len(v)>=modelLen):
            # Fit for 0:modelLen datapoints
            v_sorted = np.sort(v, order=[x])
            param_vals = param_dict[k]
            param_vals[y]=v_sorted[y][-1]
            params = {'params': model.make_params(**param_vals), x: v_sorted[-modelLen:][x]}
            fit = model.fit(v_sorted[-modelLen:][y], **params)
            
            ri = np.zeros((1,), dtype=result.dtype)
            for desc in data_desc:
                ri[0][desc[0]] = v[0][desc[0]]
            for name in model.param_names:
                ri[0][name] = fit.best_values[name]

            y_xmax = v_sorted[-1][y]
            ri[0][y] = ( y_xmax + fit.best_values[y])/2
            dEfit=( y_xmax-fit.best_values[y])/2
            dEsys=epsilon*fit.best_values[y]
            
            ri[0]['d'+y]=np.sqrt(dEfit*dEfit+dEsys*dEsys)
            ri[0]['d'+y+'fit']=fit.params[y].stderr

            result = np.concatenate((ri, result))
    return np.sort(result)

def reduceFitData(data, model, x, y, param_dict):
    min_len = np.size(model.param_names)
    data_desc = [t for t in data[list(data.keys())[0]].dtype.descr if t[0] not in [x,y]]
    dt = np.dtype( data_desc + list(map(lambda x: (x,'<f8'), model.param_names)) + [('d'+y+'fit','<f8')] )

    result = np.array([], dt )

    for k,v in data.items():
        if(len(v)>=min_len):
            params = {'params': model.make_params(**param_dict), x: v[x]}
            fit = model.fit(v[y], **params)
            
            ri = np.zeros((1,), dtype=result.dtype)
            for desc in data_desc:
                ri[0][desc[0]] = v[0][desc[0]]
            for name in model.param_names:
                ri[0][name] = fit.best_values[name]
            ri[0]['d'+y+'fit']=fit.params[y].stderr
            result = np.concatenate((ri, result))
    return np.sort(result)

currDir = os.path.dirname(os.path.realpath(__file__))
dataDir = currDir + "/"

energies = np.sort(np.loadtxt( dataDir+"energies.txt",  ndmin=1, dtype=np.dtype([('D', '<i4'), ('L', '<i4'), ('g', '<f8'), ('r', '<f8'), ('x', '<f8'), ('y', '<f8'), ('E', '<f8'), ('var', '<f8')])), order=['L', 'g', 'r', 'D'])


param_dict = defaultdict(lambda: {'E':-10000,'F1':0})
param_dict[('200.00000000','0.11000000','0.80000000')]['F1']=-0.1

modelD = models.ExpressionModel("E + F1/D",         independent_vars=['D'])
modelL = models.ExpressionModel("w0 + A/L",         independent_vars=['L'])
modelE = models.ExpressionModel("a*g*g + b*g + E0", independent_vars=['g'] )

reducedEL = reduceFitDataE(groupBy(energies, ['L', 'g', 'r']), modelD, 'D', 'E', param_dict)

energiesEL1 = mapData(reducedEL, 'w0', lambda r: r['E']/(2*r['x']*r['L']))
energiesEL = mapData(energiesEL1, 'dw0', lambda r: r['dE']/(2*r['x']*r['L']))

reducedE = reduceFitData(groupBy(energiesEL, ['g', 'r']), modelL, 'L', 'w0', {'w0':-0.6,'A':1})

data_08 = reducedE[np.where(np.isclose(reducedE['r'],0.8)) ]


fitE1_08 = modelE.fit(data_08['w0'],     g=data_08['g'],      a=2, b=1,E0=0)
fitE2_08 = modelE.fit(data_08[:-1]['w0'],g=data_08[:-1]['g'], a=2, b=1,E0=0)

csvE = np.zeros((1,), np.dtype([('r', '<f8'), ('w', '<f8'), ('dw', '<f8'), ('de', '<f8')]) )
csvE['r'] = data_08[0]['r']
csvE[0]['w']  = fitE1_08.params['E0'].value
csvE[0]['de'] = 2/np.pi + fitE1_08.params['E0'].value

dw_fit = fitE1_08.params['E0'].stderr
dw_sys = fitE2_08.params['E0'].value-fitE1_08.params['E0'].value
csvE[0]['dw'] = np.sqrt(dw_fit*dw_fit+dw_sys*dw_sys)


np.savetxt(dataDir+"energy.csv", csvE, header=" m/g w dw de")

figE, axE = plt.subplots()

locx = plticker.MultipleLocator(base=0.002)
locy = plticker.MultipleLocator(base=0.001)

axE.xaxis.set_minor_locator(locx)
axE.yaxis.set_minor_locator(locy)
axE.grid(True)

axE.set(xlabel=r"$\hat{g}$", ylabel=r"$w_0$")
axE.scatter(x=data_08['g'], y=data_08['w0'], marker='o', c='c')

ag_range = np.arange(0.07, 0.16, 0.001)
fE1 = fitE1_08.best_values['a']*(ag_range*ag_range) + fitE1_08.best_values['b']*ag_range + fitE1_08.best_values['E0']
fE2 = fitE2_08.best_values['a']*(ag_range*ag_range) + fitE2_08.best_values['b']*ag_range + fitE2_08.best_values['E0']

axE.plot(ag_range, fE2, linestyle='--', color='r')
axE.plot(ag_range, fE1)

plt.savefig(dataDir+'img/energy_08_fit.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)