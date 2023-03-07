import scipy.stats as stat
import math
import numpy as np
from _ast import arg
from NodeCluster import *

class CachedCalcItem:
    def __init__(self, p, COS_VAL, SIN_VAL):
        q1 = 1.0 - p + p * COS_VAL
        q2 = p * SIN_VAL
        self._modulus = np.sqrt(q1 * q1 + q2 * q2)
        self._arg = np.arctan2(q2, q1)
        self._log_modulus = np.log(self._modulus)

class CalcCache:
    def __init__(self, nr_of_probabilities):
        self._nr_of_probabilities = nr_of_probabilities
        self._omega = 2 * np.pi / (nr_of_probabilities + 1)
        n = int(np.ceil(nr_of_probabilities / 2)) + 1
        r = range(n)
        self.COS = [np.cos(self._omega * l) for l in r]
        self.SIN = [np.sin(self._omega * l) for l in r]
        self._arr = [dict() for l in r]
        
    def nr_of_probabilities(self):
        return self._nr_of_probabilities
    
    def getCalcItem(self, p, l):
        d = self._arr[l]
        calcItem = d.get(p)
        if (calcItem == None):
            calcItem = CachedCalcItem(p, self.COS[l], self.SIN[l])
            d[p] = calcItem
        return calcItem
        

class PBDistribution:
    def __init__(self, parray):
        self._parray = parray
        self._count = None
        if parray != None:
            self._count = len(self._parray)
        
    def parray(self):
        return self._parray
    
    def prob(self, k):
        pass
    
    def count(self):
        return self._count
     
    

                                 
       
class PBDistributionDFTCF(PBDistribution):
    def __init__(self, parray, clusters, calcCalche):
        PBDistribution.__init__(self, parray)
        self._calcCalche = None
        if calcCalche != None and calcCalche.nr_of_probabilities() == len(parray):
            self._calcCalche = calcCalche
        else:
            self._calcCalche = CalcCache(len(parray))
            
        idft_unweighted = PBDistributionDFTCF.compute_idft_unweighted(parray, clusters,  self._calcCalche)
        n = len(idft_unweighted)
        w = 1.0 / n
        idft_weighted = [w * x for x in idft_unweighted]
        self._p = np.fft.fft(idft_weighted)
        
  
    def compute_idft_unweighted_l(parray, ret, l, clusters, calcCalche):
        
        n = len(parray) - 1
        th = np.ceil(n / 2)
        omega = 2 * np.pi / n
        if l == 0:
            ret.append(np.complex(1,0))
        elif l <= th:
            sum_of_arg = 0.0
            sum_of_log_modulus = 0.0
            if clusters != None:
                for cluster in clusters:
                    cluster_size = cluster.size()
                    if cluster_size < 1:
                        continue
                    representative_node = cluster.representative_node()
                    p = parray[representative_node]
                    q1 = 1.0 - p + p*calcCalche.COS[l]
                    q2 = p * calcCalche.SIN[l]
                    modulus = np.sqrt(q1 * q1 + q2 * q2)
                    arg = np.arctan2(q2, q1)
                    sum_of_arg = sum_of_arg + arg*cluster_size
                    sum_of_log_modulus = sum_of_log_modulus + np.log(modulus) * cluster_size
                    
            else:
                for p in parray:
                    q1 = 1.0 - p + p*calcCalche.COS[l]
                    q2 = p * calcCalche.SIN[l]
                    modulus = np.sqrt(q1 * q1 + q2 * q2)
                    arg = np.arctan2(q2, q1)
                    sum_of_arg = sum_of_arg + arg
                    sum_of_log_modulus = sum_of_log_modulus + np.log(modulus)
                    
                    
            d = np.exp(sum_of_log_modulus)
            a = d * np.cos(sum_of_arg)
            b = d * np.sin(sum_of_arg)
            v = np.complex(a , b)
            ret.append(v)
        else:
            idx = n + 1 - l
            c = ret[idx]
            a = c.real
            b = c.imag
            ret.append(complex(a, -b))
            
   
      
    def compute_idft_unweighted(parray, clusters, calcCalche):
        n = len(parray)
        x = []
        r = range(0, n)
        for l in r:
            PBDistributionDFTCF.compute_idft_unweighted_l(parray,x, l, clusters, calcCalche )
        return x
   

class PBDistributionEstimatior(PBDistribution):
    def init_with_parray(self, parray):
        pass
    
    def init_with_cluster(self, cluster):
        pass      
    
    
class PBDistributionGaussEstimation(PBDistributionEstimatior):
    def __init__(self):
        PBDistribution.__init__(self, None)
        self._mean = None
        self._variance = None
        self._approximator = None
        
       
    def init_with_parray(self, parray):
        self._parray = parray
        self._mean = sum(parray)
        self._count = len(parray)
        variances = [p*(1.0 - p) for p in parray]
        self._variance = sum(variances)
        self._approximator = stat.norm(self._mean, math.sqrt(self._variance))
        
    def init_with_cluster(self, cluster):
        clusterDescriptionData = cluster.clusterDescriptionData()
        self._mean = clusterDescriptionData[NodeCluster.MEAN]
        self._variance = clusterDescriptionData[NodeCluster.VARIANCE]
        self._count = clusterDescriptionData[NodeCluster.SIZE] 
        self._approximator = stat.norm(self._mean, math.sqrt(self._variance))
    
    def prob(self, k):
        n = self.count()
        if k < 0 or k > n:
            return 0.0
        p = self._approximator.cdf(float(k) + 0.5) - self._approximator.cdf(float(k) - 0.5)
        return p
    
    
class PBDistributionPoissonEstimation(PBDistributionEstimatior):
    def __init__(self):
        PBDistribution.__init__(self, None)
        self._mean = None
        self._approximator = None
       
    def init_with_parray(self, parray):
        self._parray = parray
        self._mean = sum(parray)
        self._count = len(parray)
        self._approximator = stat.poisson(self._mean)
        
    def init_with_cluster(self, cluster):
        clusterDescriptionData = cluster.clusterDescriptionData()
        self._mean = clusterDescriptionData[NodeCluster.MEAN]
        self._count = clusterDescriptionData[NodeCluster.SIZE] 
        self._approximator = stat.poisson(self._mean)
        
    def prob(self, k):
        n = self.count()
        if k < 0 or k > n:
            return 0.0
        p = self._approximator.pmf(k)
        return p
    
class PBDistributionBinomialEstimation(PBDistributionEstimatior):
    def __init__(self):
        PBDistribution.__init__(self, None)
        self._n = None
        self._p = None
        
       
    def init_with_parray(self, parray):
        self._parray = parray
        mean = sum(parray)
        self._n = len(parray)
        self._p = mean / self._n
        self._count = len(parray)
        self._approximator = stat.binom(self._n, self._p)
        
    def init_with_cluster(self, cluster):
        clusterDescriptionData = cluster.clusterDescriptionData()
        self._n = clusterDescriptionData[NodeCluster.SIZE]
        mean = clusterDescriptionData[NodeCluster.MEAN]
        self._count = clusterDescriptionData[NodeCluster.SIZE]
        self._p = mean / self._n
        self._approximator = stat.binom(self._n, self._p)
    
    def prob(self, k):
        n = self.count()
        if k < 0 or k > n:
            return 0.0
        p = self._approximator.pmf(k)
        return p
    
class PBDistributionTranslatedPoissonEstimation(PBDistributionEstimatior):
    def __init__(self):
        PBDistribution.__init__(self, None)
        self._mean = None
        self._variance = None
        self._approximator = None
       
    def init_with_parray(self, parray):
        self._parray = parray
        self._mean = sum(parray)
        variances = [p*(1.0 - p) for p in parray]
        self._variance = sum(variances)
        d = self._mean - self._variance
        fraction_d = d - int(d)
        self._param = self._variance + fraction_d
        self._shift = -self._mean + self._variance + fraction_d 
        self._approximator = stat.poisson(self._param)
        
    def init_with_cluster(self, cluster):
        clusterDescriptionData = cluster.clusterDescriptionData()
        self._mean = clusterDescriptionData[NodeCluster.MEAN]
        self._variance = clusterDescriptionData[NodeCluster.VARIANCE]
        d = self._mean - self._variance
        fraction_d = d - int(d)
        self._param = self._variance + fraction_d
        self._shift = -self._mean + self._variance + fraction_d 
        self._approximator = stat.poisson(self._param)
    
    def prob(self, k):
        n = self.count()
        if k < 0 or k > n:
            return 0.0
        p = self._approximator.pmf(k + self._shift)
        return p