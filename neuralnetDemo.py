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
minErr = 1e-3
maxIter = 1000000
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
f = plt.figure()
plt.title('Neural Net Learns NH3 Saturation Curve')
plt.ylim(-100.,1900.)
ax1 = f.add_subplot(111)
ax1.plot(X*xNorm,Y*yNorm,'ko',label='Raw Data')
net, = ax1.plot(X*xNorm,bpn.Run(X)*yNorm,'r-',label='Neural Net')
plt.xlabel('Temperature (R)')
plt.ylabel('Pressure (PSIG)')
plt.grid()
ax1.legend(loc='best')

# Define animation
def animate(v):
    err = 0.
    for i in range(100):
        err = bpn.TrainEpoch(X,Y)
    print('Error: {0}'.format(err))
    net.set_ydata(bpn.Run(X)*yNorm)
    return net
def init():
    net.set_ydata(bpn.Run(X)*yNorm)
    return net

# animate the function
ani = animation.FuncAnimation(f,animate,init_func=init,interval=100)
plt.show()    
