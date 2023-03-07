import numpy as np
import itertools
import math
"""
This class represents an inhomogeneous random graph
"""    

class IRG:
    def __init__(self, mx):
        self._mx = mx
    
    def mx(self):
        return self._mx
       
    def expected_degree_of_node(self, node_index):
        row = self._mx[node_index]
        e = sum(row)
        return e
    
    def variance_of_node_degree(self, node_index):
        row = self._mx[node_index]
        variances = [p*(1.0 - p) for p in row]
        v = sum(variances)
        return v
    
    def expected_degree_of_nodes(self):
        n = len(self._mx[0])
        e = [self.expected_degree_of_node(i) for i in range(n)]
        return e
    