# NeuralNetworkToConstraints
Convert a neural network to a set of constraints (https://arxiv.org/abs/1712.06174)

The goal here is: Given a ReLU neural network, convert it to a set of constraints to be used in a MILP formulation.

The inputs to the converter are based on a neural net produced from sklearn, but it only requires weights and biases in that format (a numpy array essentially).

The defined object will store the constraints and variables that define the neural net. A simple example is provided to demonstrate how it could be used (its not all encompassing, but should be good enough for someone with casual interest).

More interested users have the ability to change Input and Output names. Also, the value of bigM is updatable.

Happy neural netting!

----
A class example where nonlinearities make solving an optimization problem difficult are pooling problems. Using the example provided by Gurobi, https://colab.research.google.com/github/Gurobi/modeling-examples/blob/master/pooling/std_pooling_gcl.ipynb#references, the file 'poolingProblemRelu.py' uses a neural net to replace the bilinear terms and solves the resulting MILP to within 2% of optimality.
