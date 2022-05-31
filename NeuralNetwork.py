# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:05:20 2014

@author: joshua
"""
#
# Imports
#
import numpy as np

#
# Transfer functions
#
def sgm(x, Derivative=False):
    if not Derivative:
        return 1.0/(1.0+np.exp(-1.0*x))
    else:
        out = sgm(x)
        return out*(1.0-out)

def linear(x, Derivative=False):
    if not Derivative:            
        return x
    else:
        return 1.0
        
def gaussian(x, Derivative=False):
    if not Derivative:
        return np.exp(-x**2.0)
    else:
        return (-2.0)*x*np.exp(-x**2.0)
def tanh(x, Derivative=False):
    if not Derivative:        
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x)**2.0
#
#  Classes
#
class BackPropagationNetwork:
    """back propogation neural network"""
    #
    # class members
    #
    layerCount = 0
    shape = None
    weights = []
    tFuncts = []
    
    #
    # class methods
    #
    def __init__(self,layerSize,layerFunctions = None):
        """Initialize Network"""
        # layer info
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize
        
        if layerFunctions is None:
            lFunct = []            
            for i in range(self.layerCount):
                if i == self.layerCount - 1:
                    lFunct.append(linear)
                else:
                    lFunct.append(sgm)
        else:
            if len(layerSize) != len(layerFunctions):
                raise ValueError("Incompatible list of transfer functions")
            elif layerFunctions[0] is not None:
                    raise ValueError("Input layer cannot have a transfer functions")
            else:
                lFunct = layerFunctions[1:]
                
        
        self.tFuncts = lFunct
        
        
        #input/output data from last Run
        self._layerInput = []
        self._layerOutput = []
        self._prevWeightDelta = []
        
        #create weight arrays
        for (l1,l2) in zip(layerSize[:-1],layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.2,size = (l2,l1+1)))
            self._prevWeightDelta.append(np.zeros((l2,l1+1)))
            

    # 
    # Run Method
    #
    def Run(self,input):
        """run network forward based on input format [in1,in2,...in_n]"""
        inCases = input.shape[0]
        
        # clear out the previous intermediate value lists
        self._layerInput = []
        self._layerOutput = []
        
        # run it
        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, inCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1, inCases])]))
                
            self._layerInput.append(layerInput)
            self._layerOutput.append(self.tFuncts[index](layerInput))
         
        return self._layerOutput[-1].T
         

    #
    # Train Epoch method
    #
    def TrainEpoch(self, input, target, trainingRate = 0.2, momentum = 0.5):
        """trains the network for one epoch"""
        
        delta = []
        inCases = input.shape[0]
        
        #run the network forward        
        self.Run(input)
        
        # calculate deltas
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                # compare to targets values
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta*self.tFuncts[index](self._layerInput[index],True))
            else:
                #compare to following layer's delta
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :]*self.tFuncts[index](self._layerInput[index], True))
                
        # compute weight deltas
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index
            
            if index == 0:
                layerOutput = np.vstack([input.T,np.ones([1, inCases])])
            else:
                layerOutput = np.vstack([self._layerOutput[index - 1], np.ones([1,self._layerOutput[index -1].shape[1]])])

            curWeightDelta = np.sum(\
                                layerOutput[None,:,:].transpose(2,0,1)*delta[delta_index][None,:,:].transpose(2,1,0)\
                                , axis = 0)            
            weightDelta = trainingRate * curWeightDelta + momentum * self._prevWeightDelta[index]
            
            self.weights[index] -= weightDelta
            
            self._prevWeightDelta[index] = weightDelta
            
            return error
#
# if run as a script create test object
#
if __name__ == "__main__":
    lFuncs = [None,sgm,linear]
    bpn = BackPropagationNetwork((1,200,1),lFuncs)
   
    # test run method
    lvInput = np.array([[0.128],[0.416],[0.704],[0.848],[0.952]])
    lvTarget = np.array([[0.550],[0.580],[0.645],[0.740],[0.805]])

    lnMax = 1000000
    lnErr = 1e-5
    for i in range(lnMax+1):
        err = bpn.TrainEpoch(lvInput,lvTarget)
        if i % 1000 == 0:
            print("Iteration {0}\t Error: {1:0.6f}".format(i,err))
        if err <= lnErr:
            print("Minimum error reached at iteration {0}".format(i))
            break
    print("Weight Matrices:\n {0}".format(bpn.weights))
    lvOutput = bpn.Run(lvInput)
    for i in range(lvInput.shape[0]):
        print("Input: {0} Output: {1}".format(lvInput[i], lvOutput[i]))
    
