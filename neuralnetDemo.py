# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:51:52 2015

@author: gallow1j
"""

import numpy as np
import NeuralNetwork as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Definitions
convert = 460.
xNorm = 800.
yNorm = 1700.


data = 'NH3Sat.csv'
skip = 1
maxIter = 300
netArch = (1,32,1)
actFuncts = [None,nn.sgm,nn.linear]

# Get data
x,y = np.loadtxt(data,dtype='float',delimiter=',',skiprows=skip,unpack=True)
X = x.reshape(x.shape[0],1)
Y = y.reshape(y.shape[0],1)
X += convert
X /= xNorm
Y /= yNorm

# Build Neural Network
bpn = nn.BackPropagationNetwork(netArch,actFuncts)

# Build Plot
fig, ax = plt.subplots(ncols=1,nrows=1)


# Define animation
def animate(num):
    err = 0.
    epoch = 100
    for i in range(100):
        err = bpn.TrainEpoch(X,Y)
    ax.clear()
    ax.grid(True)
    ax.set_title(f'Neural Net Learns NH3 Saturation Curve\nError: {err:0.5f}, Epoch: {num*epoch}')
    ax.plot(X*xNorm,Y*yNorm,'ko',label='Raw Data')
    ax.plot(X*xNorm,bpn.Run(X)*yNorm,'r-',label='Neural Net')

    ax.set_xlabel('Temperature (R)')
    ax.set_ylabel('Pressure (PSIG)')

    ax.set_ylim(-100.,1900.)
    ax.legend(loc='best')


# animate the function
anim = animation.FuncAnimation(fig,animate,interval=100,frames=maxIter,repeat=False)
plt.show()    
