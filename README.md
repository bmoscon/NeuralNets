NeuralNets
==========

Just some different neural network stuff I've played around with in python. 


The first thing I've posted is a feed-forward backpropagating network.

Error backpropagation is used to make small changes to the weights, according to the following formula:

![alt text](http://bit.ly/17T6Xau "backpropagation equation")


This very simple script shows the outputs of the neural network as it attempts to find the equation used to generate the test/trainig data. The graph updates itself every 500 iterations or so, so you can watch as the equation is fit by the network. 

Play with the parameters in the network to see how it affects the performance. I'll try and add more comments and add more information to the graph. The top graph is the RMSE, the 2nd graph is the training data, test data, and the current model from the neural net. The 3rd graph is of the values from the hidden outputs in the network.


This is 100% a work in progress. It needs to be cleaned up (which I might get to at some point). Most everything is hardcoded, which also needs to be changed. 


Please note that this software makes use of the numpy and matplotlib python libraries which must be installed.
