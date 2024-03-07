import sys
sys.path.append('./')
import torch
import torch.utils.data as Data
from Constant import Constant as C
from Data.Readdata import DataReader, NetDataReader
from Data.PTALKTDataset import PTALKTDataset, PTALKTNetDataset

def getTrainLoader(train_data_path, train_Net_data):
    handle = DataReader(train_data_path ,C.MAX_STEP, C.NUM_OF_QUESTIONS)
    trainques, trainans = handle.getTrainData()
    dtrain = PTALKTDataset(trainques, trainans)
    trainLoader = Data.DataLoader(dtrain, batch_size=C.BATCH_SIZE, shuffle=True)

    Net_handle = NetDataReader(train_Net_data ,C.MAX_STEP)
    trainnet = Net_handle.getTrainData()
    Net_dtrain = PTALKTNetDataset(trainnet)
    Net_trainLoader = Data.DataLoader(Net_dtrain, batch_size=C.BATCH_SIZE, shuffle=True)

    return trainLoader, Net_trainLoader

def getTestLoader(test_data_path, test_Net_data):
    handle = DataReader(test_data_path, C.MAX_STEP, C.NUM_OF_QUESTIONS)
    testques, testans = handle.getTestData()
    dtest = PTALKTDataset(testques, testans)
    testLoader = Data.DataLoader(dtest, batch_size=C.BATCH_SIZE, shuffle=False)

    Net_handle = NetDataReader(test_Net_data, C.MAX_STEP)
    testnet = Net_handle.getTestData()
    Net_dtest = PTALKTNetDataset(testnet)
    Net_testLoader = Data.DataLoader(Net_dtest, batch_size=C.BATCH_SIZE, shuffle=False)

    return testLoader, Net_testLoader


def getLoader(dataset):
    trainLoaders = []
    testLoaders = []
    Net_trainLoaders = []
    Net_testLoaders = []
    if dataset == 'assist0910':
        trainLoader, Net_trainLoader = getTrainLoader(C.Dpath + '/assist0910/assist0910_train.csv',
                                                      C.Dpath + '/assist0910/0910_Net_train.txt')
        trainLoaders.append(trainLoader)
        Net_trainLoaders.append(Net_trainLoader)

        testLoader, Net_testLoader = getTestLoader(C.Dpath + '/assist0910/assist0910_test.csv',
                                                   C.Dpath + '/assist0910/0910_Net_test.txt')
        testLoaders.append(testLoader)
        Net_testLoaders.append(Net_testLoader)

    elif dataset == 'assist2017':
        trainLoader, Net_trainLoader = getTrainLoader(C.Dpath + '/assist2017/assist2017_train.csv',
                                                      C.Dpath + '/assist2017/2017_Net_train.txt')
        trainLoaders.append(trainLoader)
        Net_trainLoaders.append(Net_trainLoader)

        testLoader, Net_testLoader = getTestLoader(C.Dpath + '/assist2017/assist2017_test.csv',
                                                   C.Dpath + '/assist2017/2017_Net_test.txt')
        testLoaders.append(testLoader)
        Net_testLoaders.append(Net_testLoader)

    elif dataset == 'Eedi':
        trainLoader, Net_trainLoader = getTrainLoader(C.Dpath + '/Eedi/Eedi_train.csv',
                                                      C.Dpath + '/Eedi/Eedi_Net_train.txt')
        trainLoaders.append(trainLoader)
        Net_trainLoaders.append(Net_trainLoader)

        testLoader, Net_testLoader = getTestLoader(C.Dpath + '/Eedi/Eedi_test.csv',
                                                    C.Dpath + '/Eedi/Eedi_Net_test.txt')
        testLoaders.append(testLoader)
        Net_testLoaders.append(Net_testLoader)

    elif dataset == 'SCD':
        trainLoader, Net_trainLoader = getTrainLoader(C.Dpath + '/SCD/SCD_train.txt',
                                     C.Dpath + '/SCD/SCD_Net_train.txt')
        trainLoaders.append(trainLoader)
        Net_trainLoaders.append(Net_trainLoader)

        testLoader, Net_testLoader = getTestLoader(C.Dpath + '/SCD/SCD_test.txt',
                                   C.Dpath + '/SCD/SCD_Net_test.txt')
        testLoaders.append(testLoader)
        Net_testLoaders.append(Net_testLoader)
        
    return trainLoaders, testLoaders, Net_trainLoaders, Net_testLoaders


    