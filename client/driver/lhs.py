import sys
import numpy as np
import json
from pyDOE import *
from scipy.stats import uniform
from hurry.filesize import size
from collections import OrderedDict


BYTES_SYSTEM = [
    (1024 ** 5, 'PB'),
    (1024 ** 4, 'TB'),
    (1024 ** 3, 'GB'),
    (1024 ** 2, 'MB'),
    (1024 ** 1, 'kB'),
    (1024 ** 0, 'B'),
]

TIME_SYSTEM = [
    (1000 * 60 * 60 * 24, 'd'),
    (1000 * 60 * 60, 'h'),
    (1000 * 60, 'min'),
    (1000, 's'),
    (1, 'ms'),
]


def get_raw_size(value, system):
    for factor, suffix in system:
        if value.endswith(suffix):
            if len(value) == len(suffix):
                amount = 1
            else:
                try:
                    amount = int(value[:-len(suffix)])
                except ValueError:
                    continue
            return amount * factor
    return None

def get_knob_raw(value, knob_type):
    if knob_type == 'integer':
        return int(value)
    elif knob_type == 'float':
        return float(value)
    elif knob_type == 'bytes':
        return get_raw_size(value, BYTES_SYSTEM)
    elif knob_type == 'time':
        return get_raw_size(value, TIME_SYSTEM)
    else:
        raise Exception('Knob Type does not support')

def get_knob_readable(value, knob_type):
    if knob_type == 'integer':
        return int(round(value))
    elif knob_type == 'float':
        return float(value)
    elif knob_type == 'bytes':
        value = int(round(value))
        return size(value, system=BYTES_SYSTEM)
    elif knob_type == 'time':
        value = int(round(value))
        return size(value, system=TIME_SYSTEM)
    else:
        raise Exception('Knob Type does not support')

def get_knobs_readable(values, types):
    result = []
    for i in range(len(values)):
        result.append(get_knob_readable(values[i], types[i]))
    return result



def main(args):
    knob_path = './knobs/postgres-96.json'
    save_path = './configs/'

    with open(knob_path, "r") as f:
        tuning_knobs = json.load(f)

    print(tuning_knobs)
    
    names = []
    maxvals = []
    minvals = []
    types = []

    for knob in tuning_knobs:
        names.append(knob['name'])
        maxvals.append(get_knob_raw(knob['tuning_range']['maxval'], knob['type']))
        minvals.append(get_knob_raw(knob['tuning_range']['minval'], knob['type']))
        types.append(knob['type'])

    #print(names, maxvals, minvals, types)

    nfeats = len(tuning_knobs)
    nsamples = 10
    samples = lhs(nfeats, samples=nsamples, criterion='maximin')
    maxvals = np.array(maxvals)
    minvals = np.array(minvals)
    scales = maxvals - minvals
    for fidx in range(nfeats):
        samples[:, fidx] = uniform(loc=minvals[fidx], scale=scales[fidx]).ppf(samples[:, fidx])

    samples_readable = []
    for sample in samples:
        samples_readable.append(get_knobs_readable(sample, types))

    print(samples_readable)

    config = {'recommendation': {}}
    for sidx in range(nsamples):
        for fidx in range(nfeats):
            config["recommendation"][names[fidx]] = samples_readable[sidx][fidx]
        with open(save_path + 'config_' + str(sidx), 'w+') as f:
            f.write(json.dumps(config))

    
if __name__ == '__main__':
    main(sys.argv)
