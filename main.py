from ResultsFolders import * 
from StaticEdgeVoting import *
from degrees import *
from os import path

"""
****************************************************************************
*******************************   The ER Test ******************************
****************************************************************************
"""

def ERTest_oneRound(n, c, resultsFolders):
    nodes = [i for i in range(0,n)]
    er_pmf = None
    p = float(c) / ( n - 1)
    network_name = "er_pmf" + str(p)
    er_pmf_csv_path = resultsFolders.createPathString(resultsFolders.irg(), network_name + ".csv")
    if path.exists(er_pmf_csv_path):
        er_pmf = readSequenceFromFile(er_pmf_csv_path)
    else:
        constDegreeSequenceGenerator = ConstDegreeSequenceGenerator(n, c)
        constDegreeSequence = constDegreeSequenceGenerator.generate()
        # With constant parameters all the edge probabilities will be equal
        # therefore we get back the Erdos-Renyi random graph
        pevm_const = ProportionalEdgeVotingModel(constDegreeSequence)
        erdos_renyi = pevm_const.generateIRG()
        
        cluster1 = NodeCluster(set(nodes), nodes[0])
        clusters = [cluster1]
        calcCache = CalcCache(n-1)
        # compute the degree distribution with DFT-CF method
        er_dftcf = IRGDegreeDistributionDFTCF(erdos_renyi, clusters, calcCache)
        er_pmf = er_dftcf.pmf()
        writeSequenceToFile(er_pmf, er_pmf_csv_path)
        
    
    
    er_pmf_plot = DegreeDistributionPlot(er_pmf)
    er_pmf_plot.plot_until(resultsFolders.createPathString(resultsFolders.irg(), network_name + "_plot.png"), int(p * 1000 * 2))
    # we know that the degree distribution must be exactly binomial
    # with parameters (n-1, c/(n-1))
   
    binom_pmf = sp.stats.binom.pmf(range(0, n), n-1, p)
    binom_pmf_plot = DegreeDistributionPlot(binom_pmf)
    binom_pmf_plot.plot_until(resultsFolders.createPathString(resultsFolders.irg(), "binom_" + str(p) +"_pmf_plot.png"), int(p * 1000 * 2))
    
    # compare er_pmf to the binomial pmf
    tv =  totalVarinceDistance(er_pmf, binom_pmf)
    return tv
    
def ERTest(n, parray, resultsFolders):
    tv_array = []
    for p in parray:
        c = p * (n - 1)
        tv = ERTest_oneRound(n, c, resultsFolders)
        tv_array.append(tv)
    return tv_array    



def executeERTest(resultsFolders):
    parray = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tv_array = ERTest(1000, parray, resultsFolders)
    print("Link probabilities: " + str(parray))
    print("TV distances: " + str(tv_array))

"""
*******************************************************************************
***********   Test the IRG degree distribution approximation method ***********
*******************************************************************************
"""

class SequenceGeneratorData:
    def __init__(self, n, mean, sd, plot_until_lognormal, plot_until_range):
        self.n = n
        self.mean = mean
        self.sd = sd
        self.plot_until_lognormal = plot_until_lognormal
        self.plot_until_range = plot_until_range
        

# creates parameters for the Voting Model to generate an IRG
def createVMParameters( degreeSequenceGenerator, plot_until, resultsFolders):
    degreeSequence = degreeSequenceGenerator.generate()
    degreeSequenceObj = DegreeSequence(degreeSequence)
    name = str(degreeSequenceGenerator)
    degreeDistrPlot = DegreeDistributionPlot(degreeSequenceObj._prob)
    plotFilePath = resultsFolders.createPathString(resultsFolders.irg(), name + ".png")
    degreeDistrPlot.plot_until(plotFilePath, plot_until)
    return degreeSequence

# calculates the exact degree distribution of the input IRG
def caclDftcfPmf(vm, name, plot_until, resultsFolders):
    pmf = None
    pmf_csv_path = resultsFolders.createPathString(resultsFolders.irg(), name + ".csv")
    if path.exists(pmf_csv_path):
        pmf = readSequenceFromFile(pmf_csv_path)
    else:
        blocks = vm.compute_blocks()
        clusters = []
        for block in blocks:
            if len(block) > 0:
                cluster = NodeCluster(set(block), block[0])
                clusters.append(cluster)
        
        irg = vm.generateIRG()
        n = len(irg) - 1
        calcCache = CalcCache(n)
        dist = IRGDegreeDistributionDFTCF(irg, clusters, calcCache)
        pmf = dist.pmf()
        writeSequenceToFile(pmf, pmf_csv_path)
        
    pmf_plot = DegreeDistributionPlot(pmf)
    pmf_plot.plot_until(resultsFolders.createPathString(resultsFolders.irg(), name + ".png"), plot_until)
    
    return pmf


""" **************************************************************************
IRG Approximation Test 1: The effect of cluster size on approximation accuracy
*************************************************************************  """


class ApproximationWithClustersResults:
    def __init__(self, cluster_size):
        self.cluster_size = cluster_size
        self.gaussian_TV = None
        self.poisson_TV = None
        self.binomial_TV = None
        
def createNodeClusters(nr_of_nodes, cluster_size):
    clusters = []
    range_start = 0
    range_end = min([cluster_size, nr_of_nodes])
    while range_end <= nr_of_nodes:
        cluster_indices = range(range_start, min([range_end, nr_of_nodes]))
        cluster = NodeCluster(set(cluster_indices), None)
        if len(cluster_indices) > 0:
            clusters.append(cluster)
        range_start = range_end
        range_end = range_end + cluster_size
    
    return(clusters)

def calcPBEstimation_withClusters(vm, pbDistributionEstimatior, name, clusters, plot_until, resultsFolders):
    pmf = None
    pmf_csv_path = resultsFolders.createPathString(resultsFolders.irg(), name + ".csv")
    if path.exists(pmf_csv_path):
        pmf = readSequenceFromFile(pmf_csv_path)
    else:
        irg = vm.generateIRG()
        dist = IRGDegreeDistributionMixureModelEstimator(irg, clusters, pbDistributionEstimatior)
        pmf = dist.pmf()
        writeSequenceToFile(pmf, pmf_csv_path)
        
    pmf_plot = DegreeDistributionPlot(pmf)
    pmf_plot.plot_until(resultsFolders.createPathString(resultsFolders.irg(), name + ".png"), plot_until)
    
    return pmf


def float_format(val):
    return "{:.6f}".format(round(val,6))

def float_format1(val):
    return "{:.1f}".format(round(val,6))

def buildClustersTestTVTable(resultsFolders, results, nr_nodes):
    filePath = resultsFolders.createPathString(resultsFolders.irg(), "ClusterTestTV.txt")
    with open(filePath, "w") as f:
        header = "Nr of clusters & cluster size & Gauss & Poisson & Binomial \\\\ \n"
        f.write(header)
        for item in results:
            cluster_size = item.cluster_size
            nr_of_clusters = int(nr_nodes / cluster_size)
            line = str(nr_of_clusters) + " & " 
            line = line + str(cluster_size) + " & "
            line = line + float_format(item.gaussian_TV) + " & "
            line = line + float_format(item.poisson_TV) + " & "
            line = line + float_format(item.binomial_TV) + " \\\\ \n"
            f.write(line)
            

def buildClustersTestTVPlot(resultsFolders, results, nr_nodes,fileName,  max_cluster_size, min_cluster_size):
    x_values = []
    x_labels = []
    gauss_TV_values = []
    poisson_TV_values = []
    binom_TV_values = []
    for item in results:
            cluster_size = item.cluster_size
            nr_of_clusters = int(nr_nodes / cluster_size)
            if cluster_size < min_cluster_size or cluster_size > max_cluster_size:
                continue
            x_values.append(nr_of_clusters)
            x_labels.append(cluster_size)
            gauss_TV_values.append(item.gaussian_TV)
            poisson_TV_values.append(item.poisson_TV)
            binom_TV_values.append(item.binomial_TV)
            
    filePath = resultsFolders.createPathString(resultsFolders.irg(), fileName)
    
    plt.plot(x_values, gauss_TV_values, color = 'r', label = "Gauss")
    plt.plot(x_values, poisson_TV_values, color = 'g', label = "Poisson")
    plt.plot(x_values, binom_TV_values, color = 'k', label = "Binomial")
    #plt.xticks(x_values, x_labels,  rotation=45)
    plt.legend()
    plt.savefig(filePath, dpi = 300)
    plt.close()      

def irgApproximationWithClustersTest(resultsFolders):
    sequenceGeneratorData =  SequenceGeneratorData(3000, 5 ,0.6, 3000, None)
    eta = 2.0
    rangeDegreeSequenceGenerator = RangeDegreeSequenceGenerator(sequenceGeneratorData.n)
    rangeDegreeSequence = createVMParameters(rangeDegreeSequenceGenerator,
                                               None,
                                               resultsFolders)
    
    vm_range = BiasedVotingModel(rangeDegreeSequence, eta)
    pmf_vm_range_name = "range_" + str(sequenceGeneratorData.n)
    pmf_vm_exact = caclDftcfPmf(vm_range,
                                pmf_vm_range_name,
                                sequenceGeneratorData.plot_until_range,
                                resultsFolders)
    cluster_sizes = [1500, 1000, 750, 600, 500, 300, 200, 100, 75, 60, 50, 30, 20, 10, 5, 1]
    results = []
    for cluster_size in cluster_sizes:
        approximation_results = ApproximationWithClustersResults(cluster_size)
        clusters = createNodeClusters(sequenceGeneratorData.n, cluster_size)
        
        pbDistributionEstimatiorGauss = PBDistributionGaussEstimation()
        pmf_gauss_name = "gauss_" + str(sequenceGeneratorData.n) + "_cluster_size_" + str(cluster_size)
        pmf_vm_gauss = calcPBEstimation_withClusters(vm_range, pbDistributionEstimatiorGauss, pmf_gauss_name, clusters, sequenceGeneratorData.plot_until_lognormal, resultsFolders)
        approximation_results.gaussian_TV = totalVarinceDistance(pmf_vm_gauss, pmf_vm_exact)
        
        pbDistributionEstimatiorPoisson = PBDistributionPoissonEstimation()
        pmf_poisson_name = "poisson_" + str(sequenceGeneratorData.n) + "_cluster_size_" + str(cluster_size)
        pmf_vm_poisson = calcPBEstimation_withClusters(vm_range, pbDistributionEstimatiorPoisson, pmf_poisson_name, clusters, sequenceGeneratorData.plot_until_lognormal, resultsFolders)
        approximation_results.poisson_TV = totalVarinceDistance(pmf_vm_poisson, pmf_vm_exact)
        
        pbDistributionEstimatiorBinom = PBDistributionBinomialEstimation()
        pmf_binom_name = "binom_" + str(sequenceGeneratorData.n) + "_cluster_size_" + str(cluster_size)
        pmf_vm_binom = calcPBEstimation_withClusters(vm_range, pbDistributionEstimatiorBinom, pmf_binom_name, clusters, sequenceGeneratorData.plot_until_lognormal, resultsFolders)
        approximation_results.binomial_TV = totalVarinceDistance(pmf_vm_binom, pmf_vm_exact)
        
        results.append(approximation_results)
    
    buildClustersTestTVTable(resultsFolders, results, sequenceGeneratorData.n)   
    buildClustersTestTVPlot(resultsFolders, results, sequenceGeneratorData.n, "ClusterTestTV_1", 1500, 30)
    buildClustersTestTVPlot(resultsFolders, results, sequenceGeneratorData.n, "ClusterTestTV_2",  100, 1)


resultsFolders = ResultsFolders("./results", "/")

""" ***************************************************************************
IRG Approximation Test 2 : The effect of network size on approximation accuracy
**************************************************************************** """

class ApproximationTestDataItem:
    def __init__(self):
        self.sequenceGeneratorData = None
        self.logNormalDegreeSequenceGenerator = None
        self.logNormalDegreeSequence = None
        self.rangeDegreeSequenceGenerator = None
        self.rangeDegreeSequence = None
        self.vm_lognormal = None
        self.pmf_lognormal_name = None
        self.vm_range = None
        self.pmf_vm_range_name = None
        self.pmf_vm_range = None
        self.pmf_lognormal_gauss_name = None
        self.pmf_lognormal_gauss = None
        self.lognormal_gauss_TV = None
        self.pmf_range_gauss_name = None
        self.pmf_range_gauss = None
        self.range_gauss_TV = None
        self.pmf_lognormal_poisson_name = None
        self.pmf_lognormal_poisson = None
        self.lognormal_poisson_TV = None
        self.pmf_range_poisson_name = None
        self.pmf_range_poisson = None
        self.range_poisson_TV = None
        self.pmf_lognormal_binom_name = None
        self.pmf_lognormal_binom = None
        self.lognormal_binom_TV = None
        self.pmf_range_binom_name = None
        self.pmf_range_binom = None
        self.range_binom_TV = None
        self.lognormal_cluster_size_statistics = None

class ClusterSizeStatistics:
    def __init__(self, n_clusters, mean_cluster_size, sd_cluster_size, min_cluster_size, max_cluster_size):
        self.n_clusters = n_clusters
        self.mean_cluster_size = mean_cluster_size
        self.sd_cluster_size = sd_cluster_size
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        

def calcPBEstimation(vm, pbDistributionEstimatior, name, plot_until, resultsFolders):
    pmf = None
    pmf_csv_path = resultsFolders.createPathString(resultsFolders.irg(), name + ".csv")
    if path.exists(pmf_csv_path):
        pmf = readSequenceFromFile(pmf_csv_path)
    else:
        blocks = vm.compute_blocks()
        clusters = []
        for block in blocks:
            if len(block) > 0:
                cluster = NodeCluster(set(block), block[0])
                clusters.append(cluster)
        
        irg = vm.generateIRG()
        dist = IRGDegreeDistributionMixureModelEstimator(irg, clusters, pbDistributionEstimatior)
        pmf = dist.pmf()
        writeSequenceToFile(pmf, pmf_csv_path)
        
    pmf_plot = DegreeDistributionPlot(pmf)
    pmf_plot.plot_until(resultsFolders.createPathString(resultsFolders.irg(), name + ".png"), plot_until)
    
    return pmf

def computeDistributions(eta, sequenceGeneratorDataList, resultsFolders):
    ret = []
    for sequenceGeneratorData in sequenceGeneratorDataList:
        apprxItem = ApproximationTestDataItem()
        ret.append(apprxItem)
        apprxItem.sequenceGeneratorData = sequenceGeneratorData
        apprxItem.logNormalDegreeSequenceGenerator = LogNormalDegreeSequenceGeneratorCDF(
                                                sequenceGeneratorData.n,
                                                sequenceGeneratorData.mean,
                                                sequenceGeneratorData.sd)
        
        apprxItem.logNormalDegreeSequence = createVMParameters(
                                                apprxItem.logNormalDegreeSequenceGenerator,
                                                None,
                                                resultsFolders)
        
        apprxItem.rangeDegreeSequenceGenerator = RangeDegreeSequenceGenerator(sequenceGeneratorData.n)
        apprxItem.rangeDegreeSequence = createVMParameters(
                                                apprxItem.rangeDegreeSequenceGenerator,
                                                None,
                                                resultsFolders)
    
        
        apprxItem.vm_lognormal = BiasedVotingModel(apprxItem.logNormalDegreeSequence, eta)
        apprxItem.pmf_lognormal_name = "lognormal_" + str(sequenceGeneratorData.n) 
        apprxItem.pmf_vm_lognormal = caclDftcfPmf(apprxItem.vm_lognormal,
                                                 apprxItem.pmf_lognormal_name,
                                                 sequenceGeneratorData.plot_until_lognormal,
                                                 resultsFolders)
        
        vm_lognormal_blocks = apprxItem.vm_lognormal.compute_blocks()
        clusters_sizes = []
        for block in vm_lognormal_blocks:
            cluster_size = len(block)
            if cluster_size > 0:
               clusters_sizes.append(cluster_size)
        
        nr_of_clusters = len(clusters_sizes)
        mean_cluster_size = np.mean(clusters_sizes)
        sd_cluster_size = np.sqrt(np.var(clusters_sizes))
        min_cluster_size = np.min(clusters_sizes)
        max_cluster_size = np.max(clusters_sizes)
        
        apprxItem.lognormal_cluster_size_statistics = ClusterSizeStatistics(nr_of_clusters, mean_cluster_size, sd_cluster_size, min_cluster_size, max_cluster_size)
        apprxItem.vm_range = BiasedVotingModel(apprxItem.rangeDegreeSequence, eta)
        apprxItem.pmf_vm_range_name = "range_" + str(sequenceGeneratorData.n)
        apprxItem.pmf_vm_range = caclDftcfPmf(apprxItem.vm_range,
                                              apprxItem.pmf_vm_range_name,
                                              sequenceGeneratorData.plot_until_range,
                                              resultsFolders)
        
        pbDistributionEstimatiorGauss = PBDistributionGaussEstimation()
        apprxItem.pmf_lognormal_gauss_name = "lognormal_" + str(sequenceGeneratorData.n) + "_gauss"
        apprxItem.pmf_lognormal_gauss = calcPBEstimation(apprxItem.vm_lognormal, pbDistributionEstimatiorGauss, apprxItem.pmf_lognormal_gauss_name,  sequenceGeneratorData.plot_until_lognormal, resultsFolders)
        apprxItem.lognormal_gauss_TV = totalVarinceDistance(apprxItem.pmf_vm_lognormal, apprxItem.pmf_lognormal_gauss)
        
        pbDistributionEstimatiorGauss = PBDistributionGaussEstimation()
        apprxItem.pmf_range_gauss_name = "range_" + str(sequenceGeneratorData.n) + "_gauss"
        apprxItem.pmf_range_gauss = calcPBEstimation(apprxItem.vm_range, pbDistributionEstimatiorGauss, apprxItem.pmf_range_gauss_name,  sequenceGeneratorData.plot_until_range, resultsFolders)
        apprxItem.range_gauss_TV = totalVarinceDistance(apprxItem.pmf_vm_range, apprxItem.pmf_range_gauss)
        
        pbDistributionEstimatiorPoisson = PBDistributionPoissonEstimation()
        apprxItem.pmf_lognormal_poisson_name = "lognormal_" + str(sequenceGeneratorData.n) + "_poisson"
        apprxItem.pmf_lognormal_poisson = calcPBEstimation(apprxItem.vm_lognormal, pbDistributionEstimatiorPoisson, apprxItem.pmf_lognormal_poisson_name,  sequenceGeneratorData.plot_until_lognormal, resultsFolders)
        apprxItem.lognormal_poisson_TV = totalVarinceDistance(apprxItem.pmf_vm_lognormal, apprxItem.pmf_lognormal_poisson)
        
        pbDistributionEstimatiorPoisson = PBDistributionPoissonEstimation()
        apprxItem.pmf_range_poisson_name = "range_" + str(sequenceGeneratorData.n) + "_poisson"
        apprxItem.pmf_range_poisson = calcPBEstimation(apprxItem.vm_range, pbDistributionEstimatiorPoisson, apprxItem.pmf_range_poisson_name,  sequenceGeneratorData.plot_until_range, resultsFolders)
        apprxItem.range_poisson_TV = totalVarinceDistance(apprxItem.pmf_vm_range, apprxItem.pmf_range_poisson)
        
        pbDistributionEstimatiorBinom = PBDistributionBinomialEstimation()
        apprxItem.pmf_lognormal_binom_name = "lognormal_" + str(sequenceGeneratorData.n) + "_binom"
        apprxItem.pmf_lognormal_binom = calcPBEstimation(apprxItem.vm_lognormal, pbDistributionEstimatiorBinom, apprxItem.pmf_lognormal_binom_name,  sequenceGeneratorData.plot_until_lognormal, resultsFolders)
        apprxItem.lognormal_binom_TV = totalVarinceDistance(apprxItem.pmf_vm_lognormal, apprxItem.pmf_lognormal_binom)
        
        pbDistributionEstimatiorBinom = PBDistributionBinomialEstimation()
        apprxItem.pmf_range_binom_name = "range_" + str(sequenceGeneratorData.n) + "_binom"
        apprxItem.pmf_range_binom = calcPBEstimation(apprxItem.vm_range, pbDistributionEstimatiorBinom, apprxItem.pmf_range_binom_name,  sequenceGeneratorData.plot_until_range, resultsFolders)
        apprxItem.range_binom_TV = totalVarinceDistance(apprxItem.pmf_vm_range, apprxItem.pmf_range_binom)
        
    return ret

def buildTotalVariationLognormalTable(approximationTestDataItemList, resultsFolders):
    filePath = resultsFolders.createPathString(resultsFolders.irg(), "lognormalTV.txt")
    with open(filePath, "w") as f:
        header = " & Gauss & Poisson & Binomial \\\\ \n"
        f.write(header)
        for item in approximationTestDataItemList:
            n = item.logNormalDegreeSequenceGenerator._n
            mean = item.logNormalDegreeSequenceGenerator._mean
            sd = item.logNormalDegreeSequenceGenerator._sd
            line = "logN(" + str(mean) + ", " + str(sd) + ", " + str(n) + ") & "
            line = line + float_format(item.lognormal_gauss_TV) + " & "
            line = line + float_format(item.lognormal_poisson_TV) + " & "
            #line = line + float_format(item.lognormal_tp_TV) + " & "
            line = line + float_format(item.lognormal_binom_TV) + " \\\\ \n"
            f.write(line)
            
            
def buildTotalVariationRangeTable(approximationTestDataItemList, resultsFolders):
    filePath = resultsFolders.createPathString(resultsFolders.irg(), "RangeTV.txt")
    with open(filePath, "w") as f:
        header = " & Gauss & Poisson & Binomial \\\\ \n"
        f.write(header)
        for item in approximationTestDataItemList:
            n = item.rangeDegreeSequenceGenerator._n
            line = "range(" + str(n) + ") & "
            line = line + float_format(item.range_gauss_TV) + " & "
            line = line + float_format(item.range_poisson_TV) + " & "
            line = line + float_format(item.range_binom_TV) + " \\\\ \n"
            f.write(line)


def buildTotalVariationLognormalPlot(approximationTestDataItemList, resultsFolders):
    x_values = []
    gauss_TV_values = []
    poisson_TV_values = []
    #TP_TV_values = []
    binom_TV_values = []
    for item in approximationTestDataItemList:
            n = item.logNormalDegreeSequenceGenerator._n
            x_values.append(n)
            gauss_TV_values.append(item.lognormal_gauss_TV)
            poisson_TV_values.append(item.lognormal_poisson_TV)
            #TP_TV_values.append(item.lognormal_tp_TV)
            binom_TV_values.append(item.lognormal_binom_TV)
    filePath = resultsFolders.createPathString(resultsFolders.irg(), "lognormalTV.png")
    plt.plot(x_values, gauss_TV_values, color = 'r', label = "Gauss")
    plt.plot(x_values, poisson_TV_values, color = 'g', label = "Poisson")
    #plt.plot(x_values, TP_TV_values, color = 'b', label = "TP")
    plt.plot(x_values, binom_TV_values, color = 'k', label = "Binomial")
    plt.legend()
    plt.savefig(filePath, dpi = 300)
    plt.close()      
    
def buildTotalVariationRangePlot(approximationTestDataItemList, resultsFolders):
    x_values = []
    gauss_TV_values = []
    poisson_TV_values = []
    #TP_TV_values = []
    binom_TV_values = []
    for item in approximationTestDataItemList:
            n = item.logNormalDegreeSequenceGenerator._n
            x_values.append(n)
            gauss_TV_values.append(item.range_gauss_TV)
            poisson_TV_values.append(item.range_poisson_TV)
            binom_TV_values.append(item.range_binom_TV)
    filePath = resultsFolders.createPathString(resultsFolders.irg(), "RangeTV.png")
    plt.plot(x_values, gauss_TV_values, color = 'r', label = "Gauss")
    plt.plot(x_values, poisson_TV_values, color = 'g', label = "Poisson")
    plt.plot(x_values, binom_TV_values, color = 'k', label = "Binomial")
    plt.legend()
    plt.savefig(filePath, dpi = 300)
    plt.close()                  

def buildLognormalSeqClusterSizeStatistics(approximationTestDataItemList, resultsFolders):
    filePath = resultsFolders.createPathString(resultsFolders.irg(), "lognormalSeqClusterSizeStatistics.txt")
    with open(filePath, "w") as f:
        header = " & nr of clusters & mean & sd & min & max  \\\\ \n"
        f.write(header)
        for item in approximationTestDataItemList:
            n = item.logNormalDegreeSequenceGenerator._n
            mean = item.logNormalDegreeSequenceGenerator._mean
            sd = item.logNormalDegreeSequenceGenerator._sd
            clusterSizeStatistics = item.lognormal_cluster_size_statistics
            line = "logN(" + str(mean) + ", " + str(sd) + ", " + str(n) + ") & "
            
            line = line + str(clusterSizeStatistics.n_clusters) + " & "
            line = line + float_format1(clusterSizeStatistics.mean_cluster_size) + " & "
            line = line + float_format1(clusterSizeStatistics.sd_cluster_size) + " & "
            line = line + str(clusterSizeStatistics.min_cluster_size) + " & "
            line = line + str(clusterSizeStatistics.max_cluster_size) + " \\\\ \n"
            f.write(line)
                                 
def irgApproximationTest(resultsFolders):
    
    sequenceGeneratorDataList = [ SequenceGeneratorData(50, 1.5 ,0.6, 20, None),
                                  SequenceGeneratorData(100, 2 ,0.6, 20, None),
                                  SequenceGeneratorData(300, 2.3 ,0.6, 20, None),
                                  SequenceGeneratorData(500, 2.7 ,0.6, 20, None),
                                  SequenceGeneratorData(800, 3 ,0.6, 30, None),
                                  SequenceGeneratorData(1000, 4 ,0.6, 70, None),
                                  SequenceGeneratorData(1500, 4.1 ,0.6, 75, None),
                                  SequenceGeneratorData(2000, 4.3 ,0.6, 100, None),
                                  SequenceGeneratorData(2500, 4.5 ,0.6, 100, None),
                                  SequenceGeneratorData(3000, 5 ,0.6, 200, None)
                                ]
    
   
    eta = 2.0
    distributions = computeDistributions(eta, sequenceGeneratorDataList, resultsFolders)
    buildTotalVariationLognormalTable(distributions, resultsFolders)
    buildTotalVariationRangeTable(distributions, resultsFolders)
    buildTotalVariationLognormalPlot(distributions, resultsFolders)
    buildTotalVariationRangePlot(distributions, resultsFolders)
    buildLognormalSeqClusterSizeStatistics(distributions, resultsFolders)

"""
****************************************************************************
*********************** Execute the test functions *************************
****************************************************************************
"""

executeERTest(resultsFolders)
irgApproximationTest(resultsFolders)
irgApproximationWithClustersTest(resultsFolders)