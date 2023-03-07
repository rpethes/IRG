import numpy as np
import scipy as sp
import scipy.stats as stat
from sympy.stats.rv import probability
from scipy import optimize
from scipy.stats import powerlaw
from scipy.stats import binom
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import csv


def writeSequenceToFile(sequence, targetFile):
    file = open(targetFile,'w')
    with file:
        writer = csv.writer(file)
        writer.writerow(sequence)


def readSequenceFromFile(sourceFile):
    sequence = list()
    with open(sourceFile) as myFile:
        reader = csv.reader(myFile)
        for row in reader:
            sequence = sequence + row
    float_sequence = [float(v) for v in sequence]
    return float_sequence


def totalVarinceDistance(p_array,q_array):
    tv = 0.0
    n = len(p_array)
    values = range(n)
    for i in values:
        tv += abs(p_array[i] - q_array[i])
    return 0.5 * tv


""" 
Base class of degree sequence generation classes
n: the length of the sequence
""" 
class DegreeSequenceGenerator:
    _n = 0
    def __init__(self,n):
        self._n = n
        
    def generate(self):
        pass
    
    def degree_sequence_from_density(self, x_array, p_array):
        degree_sequence = list()
        free_space = self._n 
        while free_space > 0: 
            for deg in x_array:
                p = p_array[deg]
                nr = min(int(round(p * self._n)),free_space)
                if (nr > 0) :
                    arr = [deg] * nr
                    degree_sequence.extend(arr)
                    free_space = free_space - nr
                if (free_space <= 0):
                    break
        return(degree_sequence)
    
    def degree_sequence_from_cdf(self, cdf):
        degree_sequence = list()
        not_finished_nodes = self._n
        max_degree = self._n - 1
        normalizer = cdf(max_degree + 0.5)
        degrees = range(0, self._n)
        for deg in degrees:
            m = deg - 0.5
            M = deg + 0.5
            p = (cdf(M) - cdf(m)) / normalizer
            nr = min(int(round(p * self._n)), not_finished_nodes)
            if nr > 0:
                arr = [deg] * nr
                degree_sequence.extend(arr)
                not_finished_nodes = not_finished_nodes - nr
            if (not_finished_nodes <= 0):
                    break
        return(degree_sequence)
                
    def __str__(self):
        return ""
"""
It generates a sequence with constant values
val: the constant value
""" 
class ConstDegreeSequenceGenerator(DegreeSequenceGenerator):
    _val = 0
    
    def __init__(self, n, val):
        DegreeSequenceGenerator.__init__(self, n)
        self._val = val
        
    def generate(self):
        degree_sequence = [self._val]*self._n
        return(degree_sequence)
    
    def __str__(self):
        s = "ConstDegreeSequenceGenerator_" + str(self._n) + "_" + str(self._val)
        return s
    
class RangeDegreeSequenceGenerator(DegreeSequenceGenerator):
    
    def __init__(self, n):
        DegreeSequenceGenerator.__init__(self, n)
        
    def generate(self):
        r = range(self._n)
        degree_sequence = [i for i in r]
        return(degree_sequence)
    
    def __str__(self):
        s = "RangeDegreeSequenceGenerator" + str(self._n) + "_" + str(self._n)
        return s

"""
Each item in the sequence is drawn independently
from a lognormal distribution with parameters mean and sd 
"""
class LogNormalDegreeSequenceGenerator(DegreeSequenceGenerator):
    _p = 0.5
    
    def __init__(self, n, mean, sd):
        DegreeSequenceGenerator.__init__(self, n)
        self._mean = mean
        self._sd = sd

        
    def generate(self):
        possible_degrees = range(self._n)
        print("lognorm_density:")
        lognorm_density = [ lognorm.pdf(degree, s = self._sd, scale = np.exp(self._mean)) for degree in possible_degrees]
        print(lognorm_density)
        degree_sequence = self.degree_sequence_from_density(possible_degrees, lognorm_density)
        return(degree_sequence)
    
    def __str__(self):
        s = "LogNormalDegreeSequenceGenerator" + str(self._n) + "_" + str(self._mean) + "_" + str(self._sd)
        return s
    
    
class LogNormalDegreeSequenceGeneratorCDF(DegreeSequenceGenerator):
    _p = 0.5
    
    def __init__(self, n, mean, sd):
        DegreeSequenceGenerator.__init__(self, n)
        self._mean = mean
        self._sd = sd

        
    def generate(self):
        cdf_lognormal = lambda x : lognorm.cdf(x,  s = self._sd, scale = np.exp(self._mean))
        degree_sequence = self.degree_sequence_from_cdf(cdf_lognormal)
        return(degree_sequence)
    
    def __str__(self):
        s = "LogNormalDegreeSequenceGenerator" + str(self._n) + "_" + str(self._mean) + "_" + str(self._sd)
        return s

    
class DegreeSequence:
    def __init__(self, degree_sequence):
        self._degree_sequence = [int(round(d)) for d in degree_sequence]
        n = len(self._degree_sequence)
        m = max(self._degree_sequence)
        
        self._frequency = [0]*(m+1)
        for d in self._degree_sequence:
            self._frequency[d] = self._frequency[d] + 1
        possible_degrees = range(m+1)
        self._prob = [ float(self._frequency[i])/n for i in possible_degrees] 
        

class DegreeDistributionPlot:
    def __init__(self, degree_distr):
        self._degree_distr = degree_distr
        
    def plot(self, target_file):
        self.plot_until(target_file, None)
        
    def plot_until(self, target_file, until_index):
        n = len(self._degree_distr)
        degrees = None
        if (until_index != None):
            n = min([n, until_index])
        degrees = range(n)
        plt.plot(degrees, self._degree_distr[0:n], 'ro')
        plt.savefig(target_file, dpi = 300)
        plt.close()
