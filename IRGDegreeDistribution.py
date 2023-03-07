from NodeCluster import *
from PBDistribution import *
import numpy as np
from pickle import NONE

class IRGDegreeDistribution:
    def __init__(self, irg_mx):
        self._irg_mx = irg_mx
        
    def pmf(self):
        pass
    
    def prob(self, k):
        probabilities = self.pmf()
        n = len(probabilities)
        if k < 0 or k > n - 1:
            return 0
        return probabilities[k]
    
    
class IRGDegreeDistributionDFTCF(IRGDegreeDistribution):
    def __init__(self, irg_mx, clusters, calcCache):
        IRGDegreeDistribution.__init__(self, irg_mx)
        if calcCache != None and calcCache.nr_of_probabilities() == (len(irg_mx) - 1):
            self._calcCache = calcCache
        else:
            self._calcCache = CalcCache(len(irg_mx) - 1)
        self._clusters = clusters
        self._p = IRGDegreeDistributionDFTCF.compute(irg_mx, clusters, self._calcCache)
    
    def compute_idft_unweighted(irg_mx, calcCache):
        nr_nodes = len(irg_mx)
        
        node_indices = range(0, nr_nodes)
        arr = None
        for node_index in node_indices:
            print(str(node_index))
            parray = irg_mx[node_index]
            idft_unweighted = PBDistributionDFTCF.compute_idft_unweighted(parray, None,  calcCache)
            if arr == None:
                arr = idft_unweighted
            else:
                arr = [sum(value) for value in zip(arr, idft_unweighted)]
                
        return arr
    
    
    def compute_idft_unweighted_with_clusters(irg_mx, clusters, calcCache):
        arr = None
        nr_of_clusters = len(clusters)
        cluster_counter = 1
        
        for cluster in clusters:
            print(str(cluster_counter) + "/" + str(nr_of_clusters))
            cluster_counter += 1
            representative_node_index = cluster.representative_node()
            parray = irg_mx[representative_node_index]
            cluster.removeRepresentativeItemAndSetNext()
            idft_unweighted_repr = PBDistributionDFTCF.compute_idft_unweighted(parray,clusters, calcCache)
            cluster.add(representative_node_index)
            n = cluster.size()
            idft_unweighted = [n * x for x in idft_unweighted_repr]
            if arr == None:
                arr = idft_unweighted
            else:
                arr = [sum(value) for value in zip(arr, idft_unweighted)]
            
        return arr
            
    def compute(irg, clusters, calcCache):
        idft_unweighted = None
        if clusters == None:
            idft_unweighted = IRGDegreeDistributionDFTCF.compute_idft_unweighted(irg, calcCache)
        else:
            idft_unweighted = IRGDegreeDistributionDFTCF.compute_idft_unweighted_with_clusters(irg, clusters, calcCache)
        
        n = len(idft_unweighted)
        w = 1.0 / (n * n)
        idft_weighted = [w * x for x in idft_unweighted]
        p = np.fft.fft(idft_weighted)
        r = range(len(p))
        ret = [p[i].real for i in r]
        return ret
        
    def pmf(self):
        return self._p   
    
    
class IRGDegreeDistributionMixureModelEstimator(IRGDegreeDistribution):
    def __init__(self, irg_mx, clusters, pbDistributionEstimatior):
        IRGDegreeDistribution.__init__(self, irg_mx)
        self._clusters = clusters
        self._pbDistributionEstimatior = pbDistributionEstimatior
        self._pmf = self.compute_pmf(irg_mx, pbDistributionEstimatior, clusters)
    
    def compute_pmf(self, irg_mx, pbDistributionEstimatior, clusters):
        n = len(irg_mx)
        possible_degrees = range(n)
        pmf = [0.0] * n
        cluster_number = len(clusters)
        cluster_index = 0
        for cluster in clusters:
            cluster_index = cluster_index + 1
            print("cluster " + str(cluster_index) + "/" + str(cluster_number))
            cluster_size = cluster.size()
            if cluster_size < 1:
                continue
            weight = float(cluster_size) / n
            representative_item = cluster.representative_node()
            if representative_item == None:
                cluster.buildClusterDescriptionDataIfNotExist(irg_mx)
                pbDistributionEstimatior.init_with_cluster(cluster)
            else:
                row = irg_mx[representative_item]
                pbDistributionEstimatior.init_with_parray(row)
            for k in possible_degrees:
                p = pbDistributionEstimatior.prob(k)
                pmf[k] += weight * p
        return pmf
    
    def pmf(self):
        return self._pmf
        