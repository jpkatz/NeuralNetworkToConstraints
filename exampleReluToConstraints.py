# -*- coding: utf-8 -*
import numpy as np
from sklearn.neural_network import MLPRegressor
import pulp,NeuralNetworkAsConstraints


#not interested in making the best neural network, just making one that works
def mimo():
    n = 20
    
    #a multiple input/output system
    x = np.linspace(-5,5,num = n)
    y = np.linspace(-5,5,num = n)
    X,Y = np.meshgrid(x,y)
    z1 = X*Y
    z2 = X**2 * Y
    
    flattenX = X.flatten()
    flattenY = Y.flatten()
    flattenZ1 = z1.flatten()
    flattenZ2 = z2.flatten()
    
    inputData = np.stack((flattenX,flattenY),axis = 1 )
    outputData = np.stack((flattenZ1,flattenZ2),axis = 1 )
    
    #'decent' size to show applicability
    nrLayers = 10,10
    
    classifier = MLPRegressor(hidden_layer_sizes=nrLayers,
                               max_iter = 500,
                               activation = 'relu', solver = 'adam',
                               random_state = 1)
    classifier.fit(inputData,outputData)
    
    pred = classifier.predict(inputData)
   
    return classifier

#creating the pulp problem - This is what would be integrated in your optimization problem
def pulpProblem(variables,constraints,initial):
    problem = pulp.LpProblem('Neural_network', pulp.LpMinimize)
    
    inputVars = [var for var in variables if var.cat == 'C' and 'Input' in var.name]
    continuousVars = [var for var in variables if var.cat == 'C' and ('Intermediate' in var.name or 'Slack' in var.name)]
    outputVars = [var for var in variables if var.cat == 'C' and 'Output' in var.name]
    binaryVars = [var for var in variables if var.cat == 'B']
        
    inVars = pulp.LpVariable.dicts('', [var.name for var in inputVars],cat = pulp.LpContinuous )
    oVars = pulp.LpVariable.dicts('', [var.name for var in outputVars],cat = pulp.LpContinuous )
    cVars = pulp.LpVariable.dicts('', [var.name for var in continuousVars],lowBound = 0, cat = pulp.LpContinuous )
    bVars = pulp.LpVariable.dicts('',[var.name for var in binaryVars],cat = pulp.LpBinary)
    
    allVars = {**inVars,**oVars,**cVars,**bVars}
    
    problem += 0, 'Empty Objective'
    
    for con in constraints:
        # pulp.constants.LpConstraintSenses
        if con.sense == '<=':
            problem += pulp.lpSum([ con.variables[var] *allVars[var.name] 
                                    for var in con.variables]) <= con.RHS, con.name
        elif con.sense == '>=':
            problem += pulp.lpSum([ con.variables[var] *allVars[var.name] 
                                    for var in con.variables]) >= con.RHS, con.name
        elif con.sense == '=':
            problem += pulp.lpSum([ con.variables[var] * allVars[var.name]
                                   for var in con.variables]) == con.RHS, con.name
    
    #initialize
    for idx,var in enumerate(inputVars):
        inVars[var.name].setInitialValue(initial[idx])
        inVars[var.name].fixValue()
      
    solution_found = problem.solve()

    
    return problem

#build classifier
classifier = mimo() 
#convert classifier to a set of constraints and variables
nnproblem = NeuralNetworkAsConstraints.NeuralNetworkConstraints(classifier.coefs_,classifier.intercepts_)

val = np.array((3,4))
classifierOutput = classifier.predict(val.reshape(1,-1))[0]

variables = [var for var in nnproblem.variables.values()]
constraints = nnproblem.constraints
problem = pulpProblem(variables,constraints,val)

idx = 0
for var in problem.variables():
    if('Output' in var.name):
        print('Output from pulp',var.varValue,'Output from neural network',classifierOutput[idx])
        idx += 1
