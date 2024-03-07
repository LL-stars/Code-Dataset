import sys
sys.path.append('./')
from Constant import Constant as C
from Model.PTALKT import PTALKT
from Data.dataloader import getTrainLoader, getTestLoader, getLoader
import torch.optim as optim
import Eval.eval as eval
import torch.nn as nn

model = PTALKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT)
optimizer_adam = optim.Adam(model.parameters(), lr=C.LR)
loss_func = eval.lossFunc()
trainLoaders, testLoaders, Net_trainLoaders, Net_testLoaders = getLoader(C.DATASET)

for epoch in range(C.EPOCH):
    print('epoch: ' + str(epoch))
    model, optimizer = eval.train(trainLoaders, model, optimizer_adam, loss_func, Net_trainLoaders)
    auc, modelx = eval.test(epoch, testLoaders, model, Net_testLoaders)
