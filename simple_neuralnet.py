'''
Copyright (C) 2010-2013  Bryant Moscon - bmoscon@gmail.com
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to 
 deal in the Software without restriction, including without limitation the 
 rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 1. Redistributions of source code must retain the above copyright notice, 
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, 
    this list of conditions and the following disclaimer in the documentation 
    and/or other materials provided with the distribution, and in the same 
    place and form as other copyright, license and disclaimer information.

 3. The end-user documentation included with the redistribution, if any, must 
    include the following acknowledgment: "This product includes software 
    developed by Bryant Moscon (http://www.bryantmoscon.org/)", in the same 
    place and form as other third-party acknowledgments. Alternately, this 
    acknowledgment may appear in the software itself, in the same form and 
    location as other such third-party acknowledgments.

 4. Except as contained in this notice, the name of the author, Bryant Moscon,
    shall not be used in advertising or otherwise to promote the sale, use or 
    other dealings in this Software without prior written authorization from 
    the author.


 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
 THE SOFTWARE.
'''

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# Make some training data
n = 20
X = np.linspace(0.,20.0,n).reshape((n,1))
T = 0.2 + 0.05 * X + 0.4 * np.sin(X) + 0.05 * np.random.normal(size=(n,1))

# Make some testing data
Xtest = X + 0.1*np.random.normal(size=(n,1))
Ttest = 0.2 + 0.05 * X + 0.4 * np.sin(Xtest) + 0.1 * np.random.normal(size=(n,1))

nSamples = X.shape[0]
nOutputs = T.shape[1]

# Set parameters of neural network
nHiddens = 20
rhoh = .9
rhoo = .1
rh = rhoh / (nSamples*nOutputs)
ro = rhoo / (nSamples*nOutputs)

# Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
V = 0.1*2*(np.random.uniform(size=(1+1,nHiddens))-0.5)
W = 0.1*2*(np.random.uniform(size=(1+nHiddens,nOutputs))-0.5)

# Add constant column of 1's
def addOnes(A):
    return np.hstack((np.ones((A.shape[0],1)),A))
X1 = addOnes(X)
Xtest1 = addOnes(Xtest)

# Take nReps steepest descent steps in gradient descent search in mean-squared-error function
nReps = 200000
# collect training and testing errors for plotting
errorTrace = np.zeros((nReps,2))
errorTrace[:] = np.nan
for reps in range(nReps):

    # Forward pass on training data
    Z = np.tanh(np.dot( X1, V ))
    Z1 = addOnes(Z)
    Y = np.dot( Z1, W )

    # Error in output
    error = Y - T
    nan = np.isnan(error)
    if nan[0]:
        print "NAN!"

    # Backward pass - the backpropagation and weight update steps
    V = V - rh * np.dot( X1.T, np.dot( error, W[1:,:].T) * (1-Z**2))
    W = W - ro * np.dot( Z1.T, error)

    # error traces for plotting
    errorTrace[reps,0] = sqrt(np.mean((error**2).flat))
    Ytest = np.dot(addOnes(np.tanh(np.dot(Xtest1,V))), W)
    errorTrace[reps,1] = sqrt(np.mean(((Ytest-Ttest)**2).flat))

    # Every so often update the graphs
    if reps % (nReps/500) == 0:
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.subplot(3,1,1)
        plt.plot(np.arange(nReps),errorTrace)

        plt.subplot(3,1,2)
        plt.plot(X,T,'o-',Xtest,Ttest,'o-',Xtest,Ytest,'o-')
        plt.legend(('Training','Testing','Model'),'lower right')
        
        plt.subplot(3,1,3)
        plt.plot(X,Z)
        plt.draw()

plt.show()
