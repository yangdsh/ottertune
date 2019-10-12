#
# OtterTune - test_ddpg.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import random
import unittest
from sklearn import datasets
import numpy as np
import torch
from analysis.ddpg.ddpg import DDPG


# test ddpg model
class TestDDPG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        super(TestDDPG, cls).setUpClass()
        boston = datasets.load_boston()
        data = boston['data']
        X_train = data[0:500]
        X_test = data[500:]
        y_train = boston['target'][0:500].reshape(500, 1)
        ddpg = DDPG(n_actions=1, n_states=13)
        for i in range(500):
            knob_data = np.array([random.random()])
            prev_metric_data = X_train[i - 1]
            metric_data = X_train[i]
            reward = y_train[i - 1]
            ddpg.add_sample(prev_metric_data, knob_data, reward, metric_data, False)
            if len(ddpg.replay_memory) > 32:
                ddpg.update()
        cls.ypreds_round = ['%.4f' % ddpg.choose_action(x)[0] for x in X_test]

    def test_ddpg_ypreds(self):  
        expected_ypreds = ['0.1770', '0.3154', '0.3056', '0.4503', '0.5706', '0.4069']
        self.assertEqual(self.ypreds_round, expected_ypreds)
