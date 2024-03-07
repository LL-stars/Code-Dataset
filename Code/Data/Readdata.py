import numpy as np
import itertools
import tqdm
import sys
sys.path.append('./')
from Constant import Constant as C
import torch

class DataReader():
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques

    def getTrainData(self):
        trainqus = torch.Tensor([])
        trainans = torch.Tensor([])
        with open(self.path, 'r') as train:
            for len, ques, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 3), desc='loading train data:    ', mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(np.int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(np.int)
                mod = 0 if len%self.maxstep == 0 else (self.maxstep - len%self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                trainqus = np.append(trainqus, ques).astype(np.int)
                trainans = np.append(trainans, ans).astype(np.int)
        return trainqus.reshape([-1, self.maxstep]), trainans.reshape([-1, self.maxstep])

    def getTestData(self):
        testqus = np.array([])
        testans = np.array([])
        with open(self.path, 'r') as test:
            for len, ques, ans in tqdm.tqdm(itertools.zip_longest(*[test] * 3), desc='loading test data:    ', mininterval=2):
                len = int(len.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                testqus = np.append(testqus, ques).astype(np.int)
                testans = np.append(testans, ans).astype(np.int)
        return testqus.reshape([-1, self.maxstep]), testans.reshape([-1, self.maxstep])
    
class NetDataReader():
    def __init__(self, path, maxstep):
        self.path = path
        self.maxstep = maxstep

    def getTrainData(self):
        trainnet = np.array([])
        with open(self.path, 'r') as Net_train:
            for net in tqdm.tqdm(itertools.zip_longest(*[Net_train]), desc='loading Net_train data:    ', mininterval=2):
                net = ','.join(net)
                # print(type(net))
                net = np.array(net.strip().strip(',').split(',')).astype(np.float).round(3)
                len1 = len(list(net))
                mod = 0 if len1%self.maxstep == 0 else (self.maxstep - len1%self.maxstep)
                zero = np.zeros(mod) - 1
                net = np.append(net, zero)
                trainnet = np.append(trainnet, net).round(3)
        return trainnet.reshape([-1, self.maxstep])


    def getTestData(self):
        testnet = np.array([])
        with open(self.path, 'r') as Net_test:
            for net in tqdm.tqdm(itertools.zip_longest(*[Net_test] * 1), desc='loading Net_test data:    ', mininterval=2):
                net = ','.join(net)
                net = np.array(net.strip().strip(',').split(',')).astype(np.float).round(3)
                len1 = len(list(net))
                mod = 0 if len1 % self.maxstep == 0 else (self.maxstep - len1 % self.maxstep)
                zero = np.zeros(mod) - 1
                net = np.append(net, zero)
                testnet = np.append(testnet, net).round(3)
        return testnet.reshape([-1, self.maxstep])