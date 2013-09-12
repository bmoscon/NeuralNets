NeuralNets
==========

Just some different neural network stuff I've played around with in python. 


The first thing I've posted is a feed-forward backpropagating network.

Error backpropagation is used to make small changes to the weights, according to the following formula:

![alt text](http://bit.ly/17T6Xau "backpropagation equation")


This very simple script shows the outputs of the neural network as it attempts to find the equation used to generate the test/trainig data. The graph updates itself every 500 iterations or so, so you can watch as the equation is fit by the network. 
