from sklearn.neighbors import KDTree
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class ConvergentCrossMapping:
  def __init__(self,source,target,dimension,tau):
    self.source = source
    self.target = target
    self.dimension = dimension
    self.tau = tau

  def ShadowManifold(self,signal):
    m = len(signal) - (self.dimension)*self.tau
    return np.array([signal[i:i + (self.dimension - 1)*self.tau + 1: self.tau] for i in range(m)])

  def GetWeights(M,s,no_neighbors=10):
    #M = Source Manifold
    #s = target state in target Manifold
    tree = KDTree(M,leaf_size=10)
    s = np.reshape(s,(1,M.shape[1]))
    dists,inds = tree.query(s,k=no_neighbors)
    dists = dists[0][1:]
    inds = inds[0][1:]
    Wi = np.exp(-dists/dists[0])
    W = Wi/np.sum(Wi)
    return inds,W

  def PredictTargets(self,ids,weights):
    neighbors = self.target[ids]
    sum=0
    for j in range(neighbors.shape[0]):
      sum = sum + weights[j]*neighbors[j]
    return sum

  def ccm(self):
    Mx = self.ShadowManifold(self.source)
    My = self.ShadowManifold(self.target)
    pred = []
    for t in range(Mx.shape[0]):
      ids,weights = self.GetWeights(Mx,Mx[t])
      p = self.PredictTargets(ids,weights)
      pred.append(p)
    return np.array(pred)
