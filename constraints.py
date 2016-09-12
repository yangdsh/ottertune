'''
Created on Sep 8, 2016

@author: dvanaken
'''

from abc import ABCMeta, abstractmethod
import numpy as np

class ConstraintHelperInterface(object):
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def apply_constraints(self, sample):
        pass

class ParamConstraintHelper(ConstraintHelperInterface):
    
    def __init__(self, params, scaler, encoder):
        if not 'inverse_transform' in dir(scaler):
            raise Exception("Scaler object must provide function inverse_transform(X)")
        if not 'transform' in dir(scaler):
            raise Exception("Scaler object must provide function transform(X)")
        self.params_ = params
        self.scaler_ = scaler
        self.encoder_ = encoder

    def apply_constraints(self, sample, scaled=True, rescale=True):
        conv_sample = self._handle_scaling(sample, scaled)

        n_values = self.encoder_.n_values
        cat_start_indices = self.encoder_.xform_start_indices
        current_idx = 0
        cat_offset = 0
        for (param, param_val) in zip(self.params_, conv_sample):
            if param.iscategorical and not param.isboolean:
                assert not param.isboolean
                assert current_idx == cat_start_indices[cat_offset]
                nvals = n_values[cat_offset]
                
                cvals = conv_sample[current_idx:current_idx+nvals]
                cvals = np.array(np.arange(nvals) == np.argmax(cvals), dtype=float)
                assert np.sum(cvals) == 1
                conv_sample[current_idx:current_idx+nvals] = cvals

                cat_offset += 1
                current_idx += nvals
            else:
                if param.isboolean:
                    pmin, pmax = 0, 1
                    param_val = round(param_val)
                else:
                    assert param.true_range is not None
                    pmin, pmax = param.true_range

                conv_sample[current_idx] = self._check_limits(param_val, pmin, pmax)
                current_idx += 1
        conv_sample = self._handle_rescaling(conv_sample, rescale)
        return conv_sample
    
    def _check_limits(self, value, vmin, vmax):
        if value > vmax:
            return vmax
        elif value < vmin:
            return vmin
        return value
    
    def _handle_scaling(self, sample, scaled):
        if scaled:
            if sample.ndim == 1:
                sample = sample.reshape(1, -1)
            sample = self.scaler_.inverse_transform(sample).ravel()
        else:
            sample = np.array(sample)
        return sample
    
    def _handle_rescaling(self, sample, rescale):
        if rescale:
            if sample.ndim == 1:
                sample = sample.reshape(1, -1)
            return self.scaler_.transform(sample).ravel()
        return sample
        
    
    def get_valid_config(self, sample, scaled=True, rescale=True):
        conv_sample = self._handle_scaling(sample, scaled)

        for i, (param, param_val) in enumerate(zip(self.params_, conv_sample)):
            if param.isinteger:
                conv_sample[i] = round(param_val)
        
        conv_sample = self.apply_constraints(conv_sample,
                                             scaled=False,
                                             rescale=False)
        
        if conv_sample.ndim == 1:
            conv_sample = conv_sample.reshape(1, -1)
        conv_sample = self.encoder_.inverse_transform(conv_sample).squeeze()

        conv_sample = self._handle_rescaling(conv_sample, rescale)
        return conv_sample
    
    def randomize_categorical_features(self, sample, scaled=True, rescale=True):
        conv_sample = self._handle_scaling(sample, scaled)
        
        current_idx = 0
        cat_idx = 0
        for param in self.params_:
            if param.iscategorical:
                if param.isboolean:
                    nvals = 1
                else:
                    assert current_idx == self.encoder_.xform_start_indices[cat_idx]
                    nvals = self.encoder_.n_values[cat_idx]
                    cat_idx += 1
                flip = np.random.binomial(1.0, 0.5)
                if flip == 1:
                    current_val = conv_sample[current_idx:current_idx+nvals]
                    #print "{}: current val={}".format(param.name, current_val)
                    assert np.all(np.logical_or(current_val == 0, current_val == 1))
                    if param.isboolean:
                        current_val = current_val.squeeze()
                        r = 1 if current_val == 0 else 0
                    else:
                        choices = np.arange(nvals)[current_val != 1]
                        assert choices.size == nvals - 1
                        r = np.zeros(nvals)
                        r[np.random.choice(choices)] = 1
                        assert np.sum(r) == 1
                    #print "{}: changing from {} to {}".format(param.name, current_val, r)
                    conv_sample[current_idx:current_idx+nvals] = r
                    
                current_idx += nvals
            else:
                current_idx += 1
        conv_sample = self._handle_rescaling(conv_sample, rescale)
        return conv_sample