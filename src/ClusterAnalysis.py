import numpy as np
import itertools as it
from scipy.special import comb

###
# [*] avg distance between positive pairs
#     [*] including standard deviation
# [*] avg distance between all elements
# [*] avg distance of furthest positive pairs
# [*] avg distance of centroids
#     est center = centroid, avg of all points of same cluster
# [*] avg cluster radius
#     radius = distance of est center to furthest positive
# [*] pct of negatives in cluster
#     num of negatives / (num of elements + num of negatives) [0-1]
# [*] pct of negatives in marginal distance
# [*] avg distance of closest negatives
#     closest to centroid
# [ ] one shot classification accuracy
#     its stochastic, so might not be the best idea to use it
###

def dist(a, b):
    if (len(a.shape) != len(b.shape)):
        raise(Exception("shape mismatch"))
    if (len(a.shape) == 2):
        return np.linalg.norm(a-b, axis=1) #L2
    return np.linalg.norm(a-b) #L2

def centroidCoordinates(A):
    return np.mean(A, axis=0)

def getCentroids(A):
    C = np.zeros((A.shape[0], A.shape[2]))
    for i in range(C.shape[0]):
        C[i, :] = centroidCoordinates(A[i,:,:])
    return C

def indexPairs(n):
    return np.array([list(j) for j in it.combinations([i for i in range(n)], 2)])

def positivePairs(k, n, lut):
    mask = lut[:,0] // n == lut[:,1] // n
    assert k * comb(n, 2, repetition=False) == len(lut[mask])
    return lut[mask]

###

def avgCentroidDistance(C, k, d):
    dc = distFromCentroids(C, np.reshape(C, (k,1,d)))
    idx = indexPairs(k)
    return np.mean(dc[idx[:,0], idx[:,1]])

def distFromCentroids(C, A):
    d = np.zeros((C.shape[0], A.shape[0]*A.shape[1]))
    for i in range(C.shape[0]):
        d[i,:] = dist(np.repeat([C[i]], A.shape[0]*A.shape[1], axis=0), np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2])))
    return d

def sameClassCentroidDistances(DFC, k, n):
    #dfc = distFromCentroids(C,A)
    mask = np.zeros(DFC.shape, dtype=int)
    for i in range(k):
        mask[i, i*n:(i+1)*n] = 1
    dfcpos = np.reshape(DFC[mask > 0], (k,n))
    return dfcpos

def diffClassCentroidDistances(DFC, k, n):
    mask = np.ones(DFC.shape, dtype=int)
    for i in range(k):
        mask[i, i*n:(i+1)*n] = 0
    dfcneg = np.reshape(DFC[mask > 0], (k,(k-1)*n))
    return dfcneg

def closestNegativesToCentroids(DFCNeg):
    return np.min(DFCNeg, axis=1)

def avgClosestNegatives(DFCNeg):
    return np.mean(closestNegativesToCentroids(DFCNeg))

def clusterRadiuses(DFCPos):
    return np.max(DFCPos, axis=1)

def avgClusterRadius(DFCPos):
    return np.mean(clusterRadiuses(DFCPos))

def numOfNegativesInCluster(DFC, radiuses, k, n, margin = 0):
    return np.array([np.sum(DFC[i,:] <= radiuses[i] + margin) - n for i in range(k)])

def pctOfNegativesInCluster(DFC, radiuses, k, n, margin = 0):
    negnums = numOfNegativesInCluster(DFC, radiuses, k, n, margin)
    return np.array([s / (s + n) for s in negnums])

def distMatrix(A):
    l = A.shape[0] * A.shape[1]
    lut = indexPairs(l)
    A_ = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
    D = dist(A_[lut[:,0]], A_[lut[:,1]])
    return D, A_, lut

def avgPositiveDistance(A_, lut, k, n):
    plut = positivePairs(k,n,lut)
    d_ = dist(A_[plut[:,0]], A_[plut[:,1]])
    avg = np.mean(d_)
    std = np.std(d_)
    return avg, std, d_

def furthestPositiveDistances(d_, k):
    #d_ = dist(A_[plut[:,0]], A_[plut[:,1]])
    d__ = np.reshape(d_, (k, d_.shape[0]//k))
    return np.max(d__, axis=1)

def avgDistanceOfFurthestPositives(d_, k):
    return np.mean(furthestPositiveDistances(d_, k))

def avgDistanceBetweenElements(D):
    return np.mean(D), np.std(D)

###

def report(A, k=None, n=None, d=None, margin=0.2):
    if (k is None or d is None or d is None):
        k,n,d = A.shape
    C = getCentroids(A)
    aCD = avgCentroidDistance(C, k, d)
    DFC = distFromCentroids(C, A)
    DFCPos = sameClassCentroidDistances(DFC, k, n)
    DFCNeg = diffClassCentroidDistances(DFC, k, n)
    aCN = avgClosestNegatives(DFCNeg)
    aCR = avgClusterRadius(DFCPos)
    radiuses = clusterRadiuses(DFCPos)
    pctNeg = pctOfNegativesInCluster(DFC, radiuses, k, n)
    pctNegwM = pctOfNegativesInCluster(DFC, radiuses, k, n, margin=margin)
    D, A_, lut = distMatrix(A)
    aPD, stdPD, d_ = avgPositiveDistance(A_, lut, k, n)
    aDFP = avgDistanceOfFurthestPositives(d_, k)
    aD,stdD = avgDistanceBetweenElements(D)
    
    return {"avgCentroidDistance": aCD,
           "avgClosestNegatives": aCN,
           "avgClusterRadius": aCR,
           #"pctOfNegativesInCluster": pctNeg,
           #"pctOfNegativesInClusterWithMargin": pctNegwM,
           "avgPctOfNegativesInCluster": np.mean(pctNeg),
           "avgPctOfNegativesInClusterWithMargin": np.mean(pctNegwM),
           "avgPositiveDistance": aPD,
           "stdPositiveDistance": stdPD,
           "avgDistanceOfFurthestPositives": aDFP,
           "avgDistanceBetweenElements": aD,
           "stdDistanceBetweenElements": stdD}