'''
Created on Jul 4, 2016

@author: dvanaken
'''

import numpy as np


NEARZERO = 1.e-8

def stdev_zero(data, axis=None):
    mstd = np.expand_dims(data.std(axis=axis), axis=axis)
    return (np.abs(mstd) < NEARZERO).squeeze()