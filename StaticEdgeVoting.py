import matplotlib.pyplot as plt
from IRG import *
import scipy.stats as stat
import math
from NodeCluster import *
from IRGDegreeDistribution import *
"""
Static edge voting model
"""
class EdgeVotingModel:
    _degreeSequence = []
    
    def __init__(self, degreeSequence):
        self._degreeSequence = degreeSequence
    
    """
    Generates the corresponding Inhomogeneous Random Graph model 
    """
    def generateIRG(self):
        pass
    
    """
    Returns the blocks of the model
    block[d] contains the nodes which desired degree is d
    """
    def compute_blocks(self):
        n = len(self._degreeSequence)
        blocks = [[] for i in range(n)]
        for i in range(n):
            d = self._degreeSequence[i]
            blocks[d].append(i)
        return blocks
    """
    computes the mean desired degree
    """
    def meanDesiredDegree(self):
        m = np.mean(self._degreeSequence)
        return m

    
    def computeDegreePmfDftCf(self, calcCache):
        blocks = self.compute_blocks()
        clusters = []
        for block in blocks:
            if len(block) > 0:
                cluster = NodeCluster(block, block[0])
                clusters.append(cluster)
        
        irg = self.generateIRG()
        dist = IRGDegreeDistributionDFTCF(irg, clusters, calcCache)
        return dist.pmf()
        
""" 
This class represents the proportional edge voting model
"""   
class ProportionalEdgeVotingModel(EdgeVotingModel):
    
    def __init__(self, degreeSequence):
        EdgeVotingModel.__init__(self, degreeSequence)
        
    def generateIRG(self):
        n = len(self._degreeSequence)
        r = range(n)
        mx = [[ (self._degreeSequence[i]+self._degreeSequence[j])/((n-1)*2.0) for i in r] for j in r]
        for i in r:
            mx[i][i] = 0.0
        return mx

    
class BiasedVotingModel(EdgeVotingModel):
    
    def __init__(self, degreeSequence, eta_):
        EdgeVotingModel.__init__(self, degreeSequence)
        self._eta = eta_
        
    def generateIRG(self):
        n = len(self._degreeSequence)
        r = range(n)
        p_array = [(float(d)/float(n-1))*(1.0 - math.exp(-self._eta * (float(d)/float(n-1)))) for d in self._degreeSequence]
        mx = [[ p_array[i] + p_array[j] - p_array[i]*p_array[j]  for i in r] for j in r]
        for i in r:
            mx[i][i] = 0.0
        return mx