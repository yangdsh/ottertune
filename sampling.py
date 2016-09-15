'''
Created on Sep 13, 2016

@author: dvanaken
'''

import numpy as np
from pyDOE import lhs
from scipy.stats.distributions import norm

class Sampler(object):
    
    def __init__(self):
        pass
    
    def gen_lhs_configs(self):
        pass

def gen_samples(n_feats, n_samples, criterion='m',
                mean=None, std=None):
    s = lhs(n_feats, samples=n_samples, criterion=criterion)
    if mean is not None:
        assert std is not None
        if np.isscalar(mean):
            assert np.isscalar(std)
            s = norm(loc=mean, scale=std).ppf(s)
        else:
            assert isinstance(mean, np.ndarray)
            assert isinstance(std, np.ndarray)
            for i in range(n_feats):
                s[:,i] = norm(loc=mean[i], scale=std[i]).ppf(s[:,i])
    return s