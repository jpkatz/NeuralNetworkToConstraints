# NeuralNetworkToConstraints
Convert a neural network to a set of constraints

The goal here is: Given a ReLU neural network, convert it to a set of constraints to be used in a MILP formulation.

The inputs to the converter are based on a neural net produced from sklearn, but it only requires weights and biases in that format (a numpy array essentially).

The defined object will store the constraints and variables that define the neural net. A simple example is provided to demonstrate how it could be used (its not all encompassing, but should be good enough for someone with casual interest).

More interested users have the ability to change Input and Output names. Also, the value of bigM is updatable.

Happy neural netting!
