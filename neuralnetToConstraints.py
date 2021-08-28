# -*- coding: utf-8 -*-

class Constraint():
    
    def __init__(self, name, sense, RHS ):
        self.name = name
        self.variables = {}
        self.RHS = RHS
        self.sense = sense
        
    def addVariable(self,variable,coef):
        self.variables[variable] = coef
        
    def removeVariable(self,variable):
        del self.variables[variable]


class Variable():
    
    def __init__(self, name = '', cat = '', LB = '', UB = ''):
        self.name = name
        self.cat = cat
        self.LB = LB
        self.UB = UB
        
    def __hash__(self):
        return hash((self.name))
    
    def __eq__(self, other):
        return self.name == other.name
    
if __name__ == '__main__':
    var = Variable('var1','C')
    con1 = Constraint('con1','<=',1)
    
    con1.addVariable(var,2)