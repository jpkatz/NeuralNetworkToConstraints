# -*- coding: utf-8 -*-

import pulp as pulp
from pulp import solvers
from sklearn.neural_network import MLPRegressor
import numpy as np
import NeuralNetworkAsConstraints

sources = ['F1','F2','F3']
pools  = ['P1','P2']
targets = ['B1','B2','B3']
content = ['DEN','BNZ','ROZ','MOZ']

class Pooling():
    
    sources = ['F1','F2','F3']
    pools  = ['P1','P2']
    targets = ['B1','B2','B3']
    content = ['DEN','BNZ','ROZ','MOZ']
    
    def __init__(self,
                 network,
                 costs,
                 prices,
                 sourceMaxBound,
                 poolMaxBound,
                 targetMinBound,
                 arcBound,
                 blendMinBounds,
                 blendMaxBounds,
                 feedContent,
                 useTightBounds,
                 t
                 ):
        self.network = network
        self.costs = costs
        self.prices = prices
        self.sourceMaxBound = sourceMaxBound
        self.poolMaxBound = poolMaxBound
        self.targetMinBound = targetMinBound
        self.arcBound = arcBound
        self.blendMinBounds = blendMinBounds
        self.blendMaxBounds = blendMaxBounds
        self.feedContent = feedContent

        self.defineProblem('PoolingFixed')
        self.defineNetwork()
        self.defineVariables()
        if(useTightBounds):
            self.setTightKnownTargetBounds(t)
        self.createObjective()
        self.createPoolConservation()
        self.createSourceBound()
        self.createDemandBound()
        self.createPoolBound()
        self.createBlending()
        self.createBlendingLowerBound()
        self.createBlendingUpperBound()
        self.createArcBound()
        
        self.netInputVars = []
        
        
        
    def defineProblem(self,name):
        self.problem = pulp.LpProblem(name,pulp.LpMaximize)

    def defineNetwork(self):     
        #create all the of the connections for the network
        #aka, create arc flows
        self.allConnections = []
        for i,value in self.network.items():
            ij = i + '-'
            for j in value:
                self.allConnections.append(ij + j)

    def defineVariables(self):
        #creating connection variables
        self.xij = pulp.LpVariable.dicts('',
                                    self.allConnections,
                                    lowBound = 0,
                                    upBound = sum(self.sourceMaxBound.values()),
                                    cat = pulp.LpContinuous
                                    )
        
        #getting better bounds
        for pool in self.pools:
            for target in self.targets:
                self.xij[pool + '-' + target].upBound = self.poolMaxBound[pool]
        
        self.q_pk = pulp.LpVariable.dicts('',
                                          [pool+'-'+attr for pool in self.pools for attr in self.content],
                                          lowBound = 0,
                                          cat = pulp.LpContinuous
                                          )
        #better bounds for quality
        for pool in self.pools:
            for attr in self.content:
                m = 0
                for feed,comp in feedContent.items():
                    tmp = comp[attr]
                    if(tmp > m):
                        m = tmp
                self.q_pk[pool + '-' + attr].upBound = m
        
        #quality * flow
        names = []
        for pool in self.pools:
            for attr in self.content:
                for target in self.targets:
                    name = pool + '-' +attr + '-' + target
                    names.append(name)
        self.w_pkt = pulp.LpVariable.dicts('',
                                          names,
                                          lowBound = 0,
                                          cat = pulp.LpContinuous
                                          )
        

    def createObjective(self):        
        #objective function    
        self.problem += pulp.lpSum([-self.costs.get(source) * self.xij[source+'-'+j] for source in self.sources for j in self.pools+self.targets if self.xij.get(source+'-'+j) is not None]
                              +[-self.prices.get(target) * self.xij[i+'-'+target] for target in self.targets for i in self.sources + self.pools if self.xij.get(i + '-' + target) is not None]
                              )
        
    def createPoolConservation(self):
        #pool conservation
        for pool in self.pools:
            self.problem += pulp.lpSum([self.xij[source + '-' + pool] for source in self.sources]
                                  + [-self.xij[pool + '-' + target] for target in self.targets]
                                  ) == 0,'Pool Conservation_'+pool

    def createSourceBound(self):
        #source bound
        for source in self.sources:
            self.problem += pulp.lpSum(self.xij[source + '-' + j] for j in self.pools+self.targets if self.xij.get(source+'-'+j) is not None
                                  ) <= self.sourceMaxBound[source],'SourceBound_'+source

    def createDemandBound(self):
        #demand bound
        for target in self.targets:
            self.problem += pulp.lpSum(self.xij[i+'-'+target] for i in self.pools+self.sources if self.xij.get(i+'-'+target) is not None
                                  ) >= self.targetMinBound[target],'DemandBound_'+target

    def createPoolBound(self):
        #pool capacity
        for pool in self.pools:
            self.problem += pulp.lpSum(self.xij[pool+'-'+j] for j in self.targets if self.xij.get(pool+'-'+j) is not None
                                  ) <= self.poolMaxBound[pool],'PoolBound_'+pool
    
    def createBlending(self):
        #blending constraint
        for pool in self.pools:
            for attr in self.content:
                self.problem += pulp.lpSum([self.w_pkt[pool+'-'+attr+'-'+target] for target in self.targets if self.xij.get(pool+'-'+target) is not None]
                                      +[-self.feedContent[source][attr] * self.xij[source+'-'+pool] for source in self.sources if self.xij.get(source+'-'+pool) is not None]
                                      ) == 0, 'Blending_'+pool+'_'+attr

    def createBlendingLowerBound(self):
        #blending lower bound
        for target in self.targets:
            for attr in self.content:
                if(self.blendMinBounds.get(target).get(attr) is not None
                   ):
                    inFromSources = [self.feedContent[source][attr]*self.xij[source+'-'+target] for source in self.sources if self.xij.get(source+'-'+target) is not None]
                    inFromPools = [self.w_pkt[pool + '-' +attr + '-' + target] for pool in self.pools if self.xij.get(pool + '-' + target) is not None]
                    total = [-self.blendMinBounds.get(target).get(attr) * self.xij[i + '-' + target] for i in self.sources + self.pools if self.xij.get(i + '-' + target) is not None]
                    self.problem += pulp.lpSum(inFromSources
                                          +inFromPools
                                          +total
                                          ) >= 0, 'BlendingLower_'+target+'_'+attr

    def createBlendingUpperBound(self):
        #blending upper bound
        for target in self.targets:
            for attr in self.content:
                if(self.blendMaxBounds.get(target).get(attr) is not None):
                    inFromSources = [self.feedContent[source][attr]*self.xij[source+'-'+target] for source in self.sources if self.xij.get(source+'-'+target) is not None]
                    inFromPools = [self.w_pkt[pool + '-' +attr + '-' + target] for pool in self.pools if self.xij.get(pool + '-' + target) is not None]
                    total = [-self.blendMaxBounds.get(target).get(attr) * self.xij[i + '-' + target] for i in self.sources + self.pools if self.xij.get(i + '-' + target) is not None]
                    self.problem += pulp.lpSum(inFromSources
                                          +inFromPools
                                          +total
                                          ) <= 0, 'BlendingUpper_'+target+'_'+attr

    def createArcBound(self):
        #arc bounds
        for ij,val in self.arcBound.items():
            self.problem += self.xij[ij] <= val,'ArcBound_'+ij
            
    def createUserDefinedConstraint(self,variables,constraints,pool,attr,target):
        #function is mostly hard coded at this time.
        #expecting input/output variables to be scaled versions
        inputVars = [var for varname,var in variables.items() if var.cat == 'C' and 'Input' in var.detail]
        continuousVars = [var for varname,var in variables.items() if var.cat == 'C' and ('Intermediate' in var.detail or 'Slack' in var.detail)]
        outputVars = [var for varname,var in variables.items() if var.cat == 'C' and 'Output' in var.detail]
        binaryVars = [var for varname,var in variables.items() if var.cat == 'B']
        
        inVars = pulp.LpVariable.dicts('', [var.name for var in inputVars if self.xij.get(var.name) is None],cat = pulp.LpContinuous )
        oVars = pulp.LpVariable.dicts('', [var.name for var in outputVars],cat = pulp.LpContinuous )
        cVars = pulp.LpVariable.dicts('', [var.name for var in continuousVars],lowBound = 0, cat = pulp.LpContinuous )
        bVars = pulp.LpVariable.dicts('',[var.name for var in binaryVars],cat = pulp.LpBinary)
        
        allVars = {**inVars,**oVars,**cVars,**bVars}
        self.xij = {**self.xij,**allVars}
        for con in constraints:
            # pulp.constants.LpConstraintSenses
            if con.sense == '<=':
                self.problem += pulp.lpSum([ con.variables[var] *self.xij[var.name] 
                                        for var in con.variables]) <= con.RHS, con.name
            elif con.sense == '>=':
                self.problem += pulp.lpSum([ con.variables[var] *self.xij[var.name] 
                                        for var in con.variables]) >= con.RHS, con.name
            elif con.sense == '=':
                self.problem += pulp.lpSum([ con.variables[var] * self.xij[var.name]
                                       for var in con.variables]) == con.RHS, con.name
                
        #handle scaling
        outputScalings = []
        for var in inputVars:
            scaledVar = self.xij[var.name]
                    
            name = var.name.split(':')[0]
            if( self.xij.get(name) is not None):
                realVar = self.xij[name]
            else:
                realVar = self.q_pk[name]
            conName = 'UnscalingInput_'+var.name+'-'+pool+'_'+attr + '_'+target
            if(self.problem.constraints.get(conName) is None):
                self.problem += realVar == scaledVar * realVar.upBound, conName
            

            outputScalings.append(realVar.upBound)
        outputScaling = 1
        for up in outputScalings:
            outputScaling *= up
        
        for var in outputVars:
            scaledVar = allVars[var.name]
            name = var.name.split(':')[0]
            realVar = self.w_pkt[name]
            self.problem += realVar == scaledVar * outputScaling, 'UnscalingOutput_'+var.name + '_ target'
               
    
    def setTightKnownTargetBounds(self,t):
        tm = (1-t)
        tp = (1+t)  
        #pool flows
        self.xij['P1-B1'].lowBound = 92.8*tm
        self.xij['P1-B1'].upBound = 92.8*tp
        
        self.xij['P1-B2'].lowBound = 990.6*tm
        self.xij['P1-B2'].upBound = 990.6*tp
        
        self.xij['P1-B3'].lowBound = 0*tm
        self.xij['P1-B3'].upBound = 0*tp
        
        
        self.xij['P2-B1'].lowBound = 1450 * tm
        self.xij['P2-B1'].upBound = 1450 * tp
        
        self.xij['P2-B2'].lowBound = 0*tm
        self.xij['P2-B2'].upBound = 0*tp
        
        self.xij['P2-B3'].lowBound = 300 * tm
        self.xij['P2-B3'].upBound = 300 * tp
        
        #inlet flows
        self.xij['F1-P1'].lowBound = 325 * tm
        self.xij['F1-P1'].upBound = 325 * tp
        
        self.xij['F1-P2'].lowBound = 1750 * tm
        self.xij['F1-P2'].upBound = 1750 * tp
        
        self.xij['F1-B2'].lowBound = 0
        self.xij['F1-B2'].upBound = 0
        
        self.xij['F2-P1'].lowBound = 258.4 * tm
        self.xij['F2-P1'].upBound = 258.4 * tp
        
        self.xij['F2-P2'].lowBound = 0
        self.xij['F2-P2'].upBound = 0
        
        self.xij['F2-B1'].lowBound = 966.7 * tm
        self.xij['F2-B1'].upBound = 966.7 * tp
        
        self.xij['F2-B3'].lowBound = 200 * tm
        self.xij['F2-B3'].upBound = 200 * tp
        
        self.xij['F3-B1'].lowBound = 0
        self.xij['F3-B1'].upBound = 0

        self.xij['F3-P1'].lowBound = 500 * tm
        self.xij['F3-P1'].upBound = 500 * tp
        

        self.xij['F3-P2'].lowBound = (0)
        self.xij['F3-P2'].upBound = (0)

    
    def solve(self):    
        self.problem.solve()
        
        
    def setInitial(self):
        self.xij['F1-P1'].setInitialValue(325)
        self.xij['F1-P1'].fixValue()
        self.xij['F1-P2'].setInitialValue(1750)
        self.xij['F1-P2'].fixValue()
        self.xij['F1-B2'].setInitialValue(0)
        self.xij['F1-B2'].fixValue()
        
        self.xij['F2-P1'].setInitialValue(258.4) #258.3
        self.xij['F2-P1'].fixValue()
        self.xij['F2-P2'].setInitialValue(0)
        self.xij['F2-P2'].fixValue()
        self.xij['F2-B1'].setInitialValue(966.7)
        self.xij['F2-B1'].fixValue()
        self.xij['F2-B3'].setInitialValue(200)
        self.xij['F2-B3'].fixValue()
        
        self.xij['F3-B1'].setInitialValue(0)
        self.xij['F3-B1'].fixValue()
        self.xij['F3-P1'].setInitialValue(500)
        self.xij['F3-P1'].fixValue()
        self.xij['F3-P2'].setInitialValue(0)
        self.xij['F3-P2'].fixValue()
        
        self.xij['P1-B1'].setInitialValue(92.8)
        self.xij['P1-B1'].fixValue()
        self.xij['P1-B2'].setInitialValue(990.6)
        self.xij['P1-B2'].fixValue()
        self.xij['P1-B3'].setInitialValue(0)
        self.xij['P1-B3'].fixValue()
        
        self.xij['P2-B1'].setInitialValue(1450)
        self.xij['P2-B1'].fixValue()
        
        self.xij['P2-B2'].setInitialValue(0)
        self.xij['P2-B2'].fixValue()
        
        self.xij['P2-B3'].setInitialValue(300)
        self.xij['P2-B3'].fixValue()
    
        #439,182.59 - gurobi optimal

    def varSolution(self):
        for var in self.problem.variables():
            print(var.getName(),var.varValue)
        print('---')
        for source in self.sources:
            print([self.xij[source+'-'+pool].getName() + ' ' + str(self.xij[source+'-'+pool].varValue) for pool in self.pools])
        print('---')
        for source in self.sources:
            print([self.xij[source+'-'+target].getName() + ' ' + str(self.xij[source+'-'+target].varValue) for target in self.targets if self.xij.get(source+'-'+target) is not None])
        print('---')
        for pool in self.pools:
            print([self.xij[pool+'-'+target].getName() + ' ' + str(self.xij[pool+'-'+target].varValue) for target in self.targets])
        print('The solution status',self.problem.status)
        print('obj',self.problem.objective.value(),'best',439182.59,'gap',(439182.59-self.problem.objective.value())/439182.59)


    def constraintResidual(self):
        tol = 1e-5
        for constraint in self.problem.constraints:
            if(self.problem.constraints[constraint].sense < 0):
                if(self.problem.constraints[constraint].value() > tol ):
                    print(self.problem.constraints[constraint].name,'violation, less than',self.problem.constraints[constraint].value())
            else:                
                if(self.problem.constraints[constraint].sense == 0
                   and not (self.problem.constraints[constraint].value() < tol
                            and self.problem.constraints[constraint].value() > -tol)
                   ):
                    print(self.problem.constraints[constraint].name,'violation, equality',self.problem.constraints[constraint].value())
                elif(self.problem.constraints[constraint].value() < -tol):
                    print(self.problem.constraints[constraint].name,'violation, greater than',self.problem.constraints[constraint].value())
        
class BilinearNet():
    
    def __init__(self):
        pass
    
    
    def buildNet(self,xBounds,yBounds,n,layerSizes):
        xL,xU = xBounds[0],xBounds[1]
        yL,yU = yBounds[0],yBounds[1]
        
        x = np.linspace(xL,xU,num = n)
        y = np.linspace(yL,yU,num = n)
        X,Y = np.meshgrid(x,y)
        z = X*Y
        
        flattenX = X.flatten()
        flattenY = Y.flatten()
        flattenZ = z.flatten()
        inputData = np.stack((flattenX,flattenY),axis = 1 )
        output = flattenZ
        
        nrLayers = layerSizes
        
        classifier = MLPRegressor(hidden_layer_sizes=nrLayers,
                                   max_iter = 500,
                                   activation = 'relu', solver = 'adam',
                                   random_state = 1)
        classifier.fit(inputData,output)
        
        self.classifier = classifier
        
        
    def netToFormulation(self,inputNames,outputNames,M,prefix):
        nnproblem = NeuralNetworkAsConstraints.NeuralNetworkConstraints(self.classifier.coefs_,
                                                                        self.classifier.intercepts_,
                                                                        prefix)
        nnproblem.setInputName(inputNames)
        nnproblem.setOutputName(outputNames)
        nnproblem.setM(M)
        nnproblem.buildProblem()
        
        self.variables = [var for var in nnproblem.variables.values()]
        self.constraints = nnproblem.constraints
        
        return nnproblem
        
        
if __name__ == '__main__':
    network = {'F1':pools + ['B2'],
               'F2':pools + ['B1','B3'],
               'F3':pools + ['B1'],
               'P1':targets,
               'P2':targets
               }

    costs = {'F1':49.2,
             'F2':62.0,
             'F3':300.0
             }
    prices = {'B1':-190,
              'B2':-230,
              'B3':-150
              }
    sourceMaxBound = {'F1':60.9756*100,
                   'F2':161.29*100,
                   'F3':5*100
                   }
    poolMaxBound = {'P1':12.5 * 100,
                 'P2':17.5*100
                 }
    targetMinBound = {'B1':500,
                      'B2':500,
                      'B3':500
                      }
    arcBound = {'F1-B2':750,
                'F3-B1':750
                }
    
    feedContent = {'F1':{'DEN':0.82,
                         'BNZ':3,
                         'ROZ':99.2,
                         'MOZ':90.5
                         },
                   'F2':{'DEN':0.62,
                         'BNZ':0,
                         'ROZ':87.9,
                         'MOZ':83.5
                       },
                   'F3':{'DEN':0.75,
                         'BNZ':0,
                         'ROZ':114,
                         'MOZ':98.7
                       }
                   }
    blendMinBounds = {'B1':{'DEN':0.74,
                            'ROZ':95,
                            'MOZ':85
                            },
                      'B2':{'DEN':0.74,
                            'ROZ':96,
                            'MOZ':88},
                      'B3':{'DEN':0.74,
                            'ROZ':91}
                      }
    blendMaxBounds = {'B1':{'DEN':0.79,
                            },
                      'B2':{'DEN':0.79,
                            'BNZ':0.9},
                      'B3':{'DEN':0.79}
                      }
    
    #fixing bilinear part - not used anymore
    quality = {'P1-DEN':0.74000369242,
               'P1-BNZ':0.90002769316,
               'P1-ROZ':103.336628819,
               'P1-MOZ':92.6156650974,
               'P2-DEN':0.82,
               'P2-BNZ':3,
               'P2-ROZ':99.2,
               'P2-MOZ':90.5,
        }
    problem = Pooling(network,
                      costs,
                      prices,
                     sourceMaxBound,
                     poolMaxBound,
                     targetMinBound,
                     arcBound,
                     blendMinBounds,
                     blendMaxBounds,
                     feedContent,
                      False,
                     0.01
                     )
    

    net = BilinearNet()
    xBounds,yBounds  = [0,1],[0,1] #normalizing to be between 0 and 1 - divide by max
    n = 100
    layerSizes =  5,5,5
    net.buildNet(xBounds, yBounds, n, layerSizes)
    print('>>>Neural network built!')
    for pool in pools:
        for attr in content:
            qualityName = pool+'-'+attr+ ':Scaled'
            for target in targets:
                prefix = pool + '_' + attr + '_' + target + '_'
                flowName = pool +'-'+target + ':Scaled'
                inputNames = [flowName,qualityName]
                outputNames = [pool + '-' +attr + '-' + target + ':Scaled']
                nnproblem = net.netToFormulation(inputNames, outputNames, 2, prefix)
                problem.createUserDefinedConstraint(nnproblem.variables, nnproblem.constraints,pool,attr,target)
            
    # problem.setInitial()
    # problem.problem.writeLP('File')
    problem.solve()
    problem.varSolution()