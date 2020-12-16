import pyBigWig
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import datasets

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve





def score_record(truth, predictions, input_digits=None):
    if input_digits is None: # bin resolution
        input_digits = 3
    scale=10**input_digits
    pos_values = np.zeros(scale + 1, dtype=np.int64)
    neg_values = np.zeros(scale + 1, dtype=np.int64)
    b = scale+1
    r = (-0.5 / scale, 1.0 + 0.5 / scale)
    all_values = np.histogram(predictions, bins=b, range=r)[0]
    print(all_values,len(predictions))
    if np.sum(all_values) != len(predictions):
        raise ValueError("invalid values in 'predictions'")
    pred_pos = predictions[truth > 0]
    pos_values = np.histogram(pred_pos, bins=b, range=r)[0]
    pred_neg = predictions[truth == 0]
    neg_values = np.histogram(pred_neg, bins=b, range=r)[0]
    return (pos_values, neg_values)

def calculate_auc(pos_values,neg_values): # auc & auprc; adapted from score2018.py
    tp = np.sum(pos_values)
    fp = np.sum(neg_values)
    tn = fn = 0
    tpr = 1
    tnr = 0
    if tp == 0 or fp == 0:
        # If either class is empty, scores are undefined.
        return (float('nan'), float('nan'))
    ppv = float(tp) / (tp + fp)
    auroc = 0
    auprc = 0
    for (n_pos, n_neg) in zip(pos_values, neg_values):
        tp -= n_pos
        fn += n_pos
        fp -= n_neg
        tn += n_neg
        tpr_prev = tpr
        tnr_prev = tnr
        ppv_prev = ppv
        tpr = float(tp) / (tp + fn)
        tnr = float(tn) / (tn + fp)
        if tp + fp > 0:
            ppv = float(tp) / (tp + fp)
        else:
            ppv = ppv_prev
        auroc += (tpr_prev - tpr) * (tnr + tnr_prev) * 0.5
        auprc += (tpr_prev - tpr) * ppv_prev
    return (auroc, auprc)




class resblock(nn.Module):
  def __init__(self):
    super(resblock,self).__init__()
    self.cov1=nn.Conv1d(32,32,3,padding=1)
    self.bn1=nn.InstanceNorm1d(32)
    self.cov2=nn.Conv1d(32,32,3,padding=1)
    self.bn2=nn.InstanceNorm1d(32)

  def forward(self,x):
    residual=x
    x=self.cov1(x)
    x=self.bn1(x)
    x=F.relu(x)
    x=self.cov2(x)
    x=self.bn2(x)
    x=x+residual
    x=F.relu(x)
    return x


class Net(nn.Module):
  def __init__(self):

    super(Net,self).__init__()
    self.cov1=nn.Conv1d(5,32,3,padding=1)
    self.bn1=nn.InstanceNorm1d(32)
    self.res1=nn.Sequential(*[resblock()]*1)
    self.cov2=nn.Conv1d(32,1,3,padding=1)
  def forward(self,x):
    x=F.relu(self.cov1(x))
    x=F.relu(self.bn1(x))
    x=self.res1(x)
    x=F.sigmoid(self.cov2(x))
    #output=F.log_softmax(x,dim=1)
    return x




net = Net()
net.load_state_dict(torch.load('./save_model_lr3_wei_3_no_meth_chr_4_20201213.pkl'))
net.eval()


#load test  data
final_label_matrix_test=np.load('test_label.npy')
final_large_matrix_test=np.load('test_data.npy')
final_large_matrix_test=torch.FloatTensor(final_large_matrix_test.astype(np.float64))
final_label_matrix_test=torch.FloatTensor(final_label_matrix_test.astype(np.int32))
test=final_large_matrix_test.permute(2,0,1)
test_label=final_label_matrix_test.permute(2,0,1)


print(test_label.shape)
test_label=np.matrix(test_label.detach().numpy()).getA1()

print(test.shape)



test_predict=net(test)

test_predict=np.matrix(test_predict.detach().numpy()).getA1()
print(test_predict.shape)
print('test_predict',np.sum(test_predict))
print('test_label',np.sum(test_label))
pos_values, neg_values=score_record(test_label, test_predict, input_digits=None)
print(pos_values)
auroc,auprc=calculate_auc(pos_values,neg_values)
with open('auc_roc_result_try.txt','a') as f:
    f.write(str(auroc.item())+'\t'+str(auprc.item())+'\n')
print(auroc,auprc)


###auprc
average_precision = average_precision_score(np.abs(test_label), test_predict)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


disp = plot_precision_recall_curve(net, test, np.abs(test_label))
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


##sample bar plot


# print(test_label[700:750])
# true_hist=plt.bar(x=np.arange(50),height=test_label[700:750])
# plt.savefig('./figure/true_one_sample_hist.png')
# plt.close()
# predict_hist=plt.bar(x=np.arange(50),height=test_predict[700:750])
# plt.savefig(f'./figure/predict_one_sample.png')
# plt.close()






##auroc

# test_label=np.abs(test_label)
# fpr, tpr, thresholds = metrics.roc_curve(test_label, test_predict)
#
#
# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
#
# plt.savefig('./figure/auroc_curve_four_chr_K562_epoch15_result.png')
# plt.close()
