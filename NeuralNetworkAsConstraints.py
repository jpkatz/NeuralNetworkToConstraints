# -*- coding: utf-8 -*-
import numpy as np
import neuralnetToConstraints as nn    


class NeuralNetworkConstraints():
    
    C = 'C'
    B = 'B'
    
    def __init__( self, weights, bias  ):
        self.weights = weights
        self.bias = bias
        self.nrInputs = self.weights[0].shape[0]
        self.nrOutputs = self.weights[-1].shape[0] #could be wrong
        self.NrLayers = len(weights) - 1 #output is not a layer
        self.Input = ['Input'+str(i) for i in range(self.nrInputs)]
        self.Intermediate= 'Intermediate'
        self.Slack = 'Slack'
        self.Binary = 'Binary'
        self.Output = ['Output'+str(i) for i in range(self.nrOutputs)]
        self.M = 1000
        
        
        #variables
        self.variables = {}
        self.constraints = []
        #for 'easy' access
        self._inputvars = {}
        self._outputvars = {}
        self._binvars = {}
        
        #call the methods to build the constraints and variables
        self.createInputVars()
        self.createOutputVars()
        self.createIntermediateVars()
        self.createIntermediateConstraints()
        self.createOutputConstraints()

    def createInputVars(self):
        stage = 0
        for i in range(self.nrInputs):
            inputName = self.Input[i]
            name = NeuralNetworkConstraints.Name(inputName,stage,i)
            var = nn.Variable(name,self.C)
            self.variables[var.name] = var
    def createOutputVars(self):
        outputWeight = self.weights[-1]
        stage = self.NrLayers + 1
        
        for idx,row in enumerate(outputWeight.transpose()):
            outputName = self.Output[idx]
            name = NeuralNetworkConstraints.Name(outputName,stage,idx)
            var = nn.Variable(name,self.C)
            self.variables[var.name] = var
        
    def createIntermediateVars(self):
        for layer in range(self.NrLayers):
            A = self.weights[layer].transpose()
            stage = layer + 1
            #shouldnt be needed to enumerate A
            for idx,row in enumerate(A):
                #continuous, slack, and binary
                name = NeuralNetworkConstraints.Name(self.Intermediate, stage, idx)
                var = nn.Variable(name,self.C,0)
                self.variables[var.name] = var
                name = NeuralNetworkConstraints.Name(self.Slack, stage, idx)
                var = nn.Variable(name,self.C,0)
                self.variables[var.name] = var
                name = NeuralNetworkConstraints.Name(self.Binary,stage,idx)
                var = nn.Variable(name,self.B)
                self.variables[var.name] = var
    
    def createIntermediateConstraints(self):
        for layer in range(self.NrLayers):
            A = self.weights[layer].transpose()
            b = self.bias[layer]
            stage = layer + 1
            for idx,row in enumerate(A):
                name = NeuralNetworkConstraints.Name('Layer',stage,idx)
                con = nn.Constraint(name, '=', b[idx])
                self.addConstraint(con)
                lhsNameIntermediate = NeuralNetworkConstraints.Name(self.Intermediate,stage,idx)
                self.addVariableToConstraint(lhsNameIntermediate, 1.0, con)
                lhsNameSlack = NeuralNetworkConstraints.Name(self.Slack,stage,idx)
                self.addVariableToConstraint(lhsNameSlack, -1.0, con)
                
                for colIdx,col in enumerate(row):
                    #getting variables from previous stage
                    if stage == 1:
                        varname = self.Input[colIdx]
                    else:
                        varname = self.Intermediate
                    varname = NeuralNetworkConstraints.Name( varname, layer, colIdx )
                    self.addVariableToConstraint(varname, -col, con)
                        
                #sandwich constraint
                binvarname = NeuralNetworkConstraints.Name(self.Binary,stage,idx)
                #for intermediate
                name = NeuralNetworkConstraints.Name('IntermediateSandwich',stage,idx)
                con = nn.Constraint(name,'<=',0)
                self.addVariableToConstraint(lhsNameIntermediate, 1.0, con)
                self.addVariableToConstraint(binvarname, -self.M, con)
                self.addConstraint(con)
                #for slack
                name = NeuralNetworkConstraints.Name('SlackSandwich',stage,idx)
                con = nn.Constraint(name,'<=',self.M)
                self.addVariableToConstraint(lhsNameSlack, 1.0, con)
                self.addVariableToConstraint(binvarname, self.M, con)
                self.addConstraint(con)
              
    def createOutputConstraints(self):
        A = self.weights[-1].transpose()
        b = self.bias[-1]
        layer = self.NrLayers
        stage = self.NrLayers + 1
        for idx,row in enumerate(A):
            name = NeuralNetworkConstraints.Name('Output',stage,idx)
            con = nn.Constraint(name, '=', b[idx])
            self.addConstraint(con)
            outputName = NeuralNetworkConstraints.Name(self.Output[idx],stage,idx)
            self.addVariableToConstraint(outputName, 1.0, con)
            
            for colIdx,col in enumerate(row):
                #getting variables from previous stage
                varname = self.Intermediate
                varname = NeuralNetworkConstraints.Name( varname,layer, colIdx )
                self.addVariableToConstraint(varname, -col, con)
            
    @staticmethod
    def Name( name, stage, idx ):
        return name + '_(' + str(stage) + ',' + str(idx) + ')'
       
    def getVariable(self,name):
        var = self.variables[name]
        return var
    
    def addVariableToConstraint(self,varname,coef,con):
        var = self.variables[varname]
        con.addVariable(var,coef)
        
    def addConstraint(self,con):
        self.constraints.append(con)
    
    #setters    
    def setInputName(self, name):
        #order should be given as used by neural net
        self.Input = name
        
    def setOutputName( self, name ):
        self.Output = name
        
    def setIntermediateName( self, name ):
        self.Intermediate = name
        
    def setM( self, M ):
        self.M = M
        
        
        
if __name__ == '__main__':
    # nnproblem = NeuralNetworkConstraints(classifier.coefs_,classifier.intercepts_)
    # for con in nnproblem.constraints:
    # print('con name',con.name)
    # for var in con.variables:
    #     print('var name/coef',var.name,con.variables[var])
    pass