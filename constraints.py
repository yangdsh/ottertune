'''
Created on Sep 8, 2016

@author: dvanaken
'''

from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import sys

class ConstraintHelperInterface(object):
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def apply_constraints(self, sample):
        pass

class ParamConstraintHelper(ConstraintHelperInterface):
    
    CATEGORICAL_TYPES = ['string', 'enumeration', 'boolean']
    
    def __init__(self, params, scaler):
        if not 'inverse_transform' in dir(scaler):
            raise Exception("Scaler object must provide function inverse_transform(X)")
        if not 'transform' in dir(scaler):
            raise Exception("Scaler object must provide function transform(X)")
        self._params = params
        self._scaler = scaler

    def apply_constraints(self, sample, scaled=True, rescale=True):
        assert sample.shape[0] == len(self._params)
        if scaled:
            conv_sample = self._scaler.inverse_transform(sample)
        else:
            conv_sample = np.array(sample)
        for i, (param, param_val) in enumerate(zip(self._params, conv_sample)):
            if param.iscategorical:
                pmin = 0
                pmax = len(param.valid_values) - 1
            else:
                assert param.true_range is not None
                pmin, pmax = param.true_range
            if param_val < pmin:
                conv_sample[i] = pmin
            elif param_val > pmax:
                conv_sample[i] = pmax
        if rescale:
            conv_sample = self._scaler.transform(conv_sample)
        return conv_sample
    
    def get_valid_sample(self, sample, scaled=True, rescale=True):
        assert sample.shape[0] == len(self._params)
        if scaled:
            conv_sample = self._scaler.inverse_transform(sample)

        for i, (param, param_val) in enumerate(zip(self._params, conv_sample)):
            if param.iscategorical or param.isinteger:
                conv_sample[i] = round(param_val)
        
        conv_sample = self.apply_constraints(conv_sample,
                                             scaled=False,
                                             rescale=False)
        if rescale:
            conv_sample = self._scaler.transform(conv_sample)
        return conv_sample