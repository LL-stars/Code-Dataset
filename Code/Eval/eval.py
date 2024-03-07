import sys
sys.path.append('./')
import tqdm
import torch
import torch.nn as nn
from sklearn import metrics
from Constant import Constant as C

def performance(epoch, ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(), prediction.detach().cpu().numpy(), pos_label=1, drop_intermediate=True)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(torch.round(ground_truth).detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    MAE = metrics.mean_absolute_error(ground_truth.detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    RMSE = metrics.mean_squared_error(ground_truth.detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())**0.5
    
    print('auc: ' + str(auc))
    print('acc: ' + str(acc))
    print('MAE: ' + str(MAE))
    print('RMSE: ' + str(RMSE))
    return auc, acc

def data(pred, batch, device):
    for student in range(pred.shape[0]):
        delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
        temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())
        index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]]).to(device)
        p = temp.gather(0, index)[0]
        a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1.)//2.)[1:]
    return p, a

class lossFunc(nn.Module):
    def __init__(self):
        super(lossFunc, self).__init__()

    def forward(self, pred, batch):
        loss = torch.Tensor([0.0])
        for student in range(pred.shape[0]):
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
            temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())
            index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]])
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1.)//2.)[1:]
            for i in range(len(p)):
                if p[i] > 0:
                    loss = loss - (a[i]*torch.log(p[i]) + (1-a[i])*torch.log(1-p[i]))
        return loss

def train_epoch(model, trainLoader, optimizer, loss_func, Net_trainLoader):
    for batch, Net_batch in tqdm.tqdm(zip(trainLoader, Net_trainLoader), desc='Training:    ', mininterval=2):
        pred = model(batch, Net_batch)
        loss = loss_func(pred, batch)
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

    return model, optimizer


def test_epoch(model, testLoader, Net_testLoader):
    gold_epoch = torch.Tensor([])
    pred_epoch = torch.Tensor([])
    for batch, Net_batch in tqdm.tqdm(zip(testLoader, Net_testLoader), desc='Testing:    ', mininterval=2):
        pred = model(batch, Net_batch)
        for student in range(pred.shape[0]):
            temp_pred = torch.Tensor([])
            temp_gold = torch.Tensor([])
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
            temp = pred[student][:C.MAX_STEP].mm(delta[0:].t())
            index = torch.LongTensor([[i for i in range(C.MAX_STEP-1)]])
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)//2)[0:]
            for i in range(len(p)):
                if p[i] > 0:
                    temp_pred = torch.cat([temp_pred,p[i:i+1]])
                    temp_gold = torch.cat([temp_gold, a[i:i+1]])
            pred_epoch = torch.cat([pred_epoch, temp_pred])
            gold_epoch = torch.cat([gold_epoch, temp_gold])
    return pred_epoch, gold_epoch


def train(trainLoaders, model, optimizer, lossFunc, Net_trainLoaders):
    for i in range(len(trainLoaders)):
        model, optimizer = train_epoch(model, trainLoaders[i], optimizer, lossFunc, Net_trainLoaders[i])
    return model, optimizer

def test(epoch, testLoaders, model, Net_testLoaders):
    ground_truth = torch.Tensor([])
    prediction = torch.Tensor([])
    for i in range(len(testLoaders)):
        pred_epoch, gold_epoch = test_epoch(model, testLoaders[i], Net_testLoaders[i])
        prediction = torch.cat([prediction, pred_epoch])
        ground_truth = torch.cat([ground_truth, gold_epoch])
    auc, _ = performance(epoch, ground_truth, prediction)
    return auc, model

