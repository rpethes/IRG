import numpy as np

class NodeCluster:
    MEAN = "MEAN"
    VARIANCE = "VARIANCE"
    SIZE = "SIZE"
    
    def __init__(self, nodes, representative_node):
        self._nodes = nodes
        self._representative_node = representative_node
        self._clusterDescriptionData = dict()
        
    def nodes(self):
        return self._nodes
    
    def representative_node(self):
        return self._representative_node
    
    def size(self):
        return len(self._nodes)
    
    def remove(self, item):
        self._nodes.remove(item)
        
    def add(self, item):
        self._nodes.add(item)
    
    
    def buildClusterDescriptionDataIfNotExist(self, irg_mx):
        if len(self._clusterDescriptionData) > 0:
            return 
        node_degree_means = []
        node_degree_variances = []
        for node_index in self._nodes:
            p_array = irg_mx[node_index]
            m = np.sum(p_array)
            var = 0.0
            for p in p_array:
                var = var + p * (1.0 - p)
            node_degree_means.append(m)
            node_degree_variances.append(var)
        cluster_mean = np.mean(node_degree_means)
        cluster_variance = np.mean(node_degree_variances)
        self._clusterDescriptionData[NodeCluster.MEAN] = cluster_mean
        self._clusterDescriptionData[NodeCluster.VARIANCE] = cluster_variance
        self._clusterDescriptionData[NodeCluster.SIZE] = len(irg_mx)
            
        
    def clusterDescriptionData(self):
        return self._clusterDescriptionData
    
    def removeRepresentativeItemAndSetNext(self):
        repr = self._representative_node
        self.remove(self._representative_node)
        if self.size() > 0:
            e = self._nodes.pop()
            self._nodes.add(e)
            self._representative_node = e
        return repr
            