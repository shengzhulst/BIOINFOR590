import pyBigWig
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
####read the TF training_label

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
TF='CEBPB'
dir_label='/data/shiting/bioinfor590_deep_learning/training_labels/study_label_part'
cell_lines='A549'
#chr11, 12,13,3
chr_name='chr11'
chr_list=['chr3','chr11','chr12','chr4']
print('start')
batch_size=256
pandas_dataframe_label=pd.read_csv(f'{dir_label}/{TF}.train.labels_{chr_name}.bed',sep='\t',header=None)
TF_cell_line_label=np.matrix(pandas_dataframe_label[pandas_dataframe_label.columns[0:4]])[::4,:]
length_of_sequence_generate=6400

######open DNA
path1='/data/shiting/bioinfor590_deep_learning/sequence/'
list_dna=['A','C','G','T']
dict_dna={}
for the_id in list_dna:
        dict_dna[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')

#######open DNASE-seq
path2='/data/shiting/bioinfor590_deep_learning/DNASE/fold_coverage_wiggles/'
DNASE_seq_bigwig=pyBigWig.open(f'{path2}DNASE.{cell_lines}.fc.signal.bigwig')

######open methylation
path3='/data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/final_whole_script/changebedbigwig/'
methylation_bigwig=pyBigWig.open(f'{path3}{cell_lines}_methylation.bw')

#########open chip-seq as label
chip_seq_TF='CTCF'
path4='/data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/chip_seq_bigwig/'
chip_seq_bigwig=pyBigWig.open(f'{path4}{chip_seq_TF}_{cell_lines}.bigwig')
chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=np.array([249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560])

chip_seq_start=0
chip_seq_end=num_bp[chr_all.index(chr_name)]
num_sequences_chip_seq=(chip_seq_end-chip_seq_start)//length_of_sequence_generate
temp_bunch_chip_seq=np.arange(num_sequences_chip_seq)
temp_bunch_chip_seq=temp_bunch_chip_seq[1:]
np.random.shuffle(temp_bunch_chip_seq)
training_index_chip_seq=temp_bunch_chip_seq

#test_index_chip_seq=temp_bunch_chip_seq[int(len(temp_bunch_chip_seq)*0.8):][1:500]


def generate_data_for_net(track,batch_size,index_part,chr_name):
	final_large_matrix=np.zeros((1,1))
    #calculate how many batch will be generated
	final_label_matrix=np.zeros((1,1))

	for i in list(index_part)[track:track+batch_size]:
		empty_output=np.zeros((length_of_sequence_generate,5))
		start=index_part[track]*length_of_sequence_generate
		end=start+length_of_sequence_generate
		final_label_input_training=np.array(chip_seq_bigwig.values(chr_name,start,end))
		final_label_input_training[final_label_input_training<0]=0


		for index,the_id in enumerate(list_dna):
			empty_output[:,index]=dict_dna[the_id].values(chr_name,start,end)
		empty_output[:,4]=DNASE_seq_bigwig.values(chr_name,start,end)
		#empty_output[:,5]=methylation_bigwig.values(chr_name,start,end)
		#empty_output[:,5][np.isnan(empty_output[:,5])]=0

		if final_large_matrix.shape[0]==1:
			final_large_matrix=empty_output.T
			final_label_matrix=final_label_input_training.T
		else:
			final_large_matrix=np.dstack((empty_output.T,final_large_matrix))
			final_label_matrix=np.dstack((final_label_input_training.T,final_label_matrix))

	return final_large_matrix,final_label_matrix

final_label_matrix_test=np.load('test_label.npy')
final_large_matrix_test=np.load('test_data.npy')
final_large_matrix_test=torch.FloatTensor(final_large_matrix_test.astype(np.float64))
final_label_matrix_test=torch.FloatTensor(final_label_matrix_test.astype(np.int32))
test=final_large_matrix_test.permute(2,0,1)
test_label=final_label_matrix_test.permute(2,0,1)
test_label=np.matrix(test_label.detach().numpy()).getA1()

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
    ##we have 200bp result, we prefer to use the mid part 200bp and around(600bp) as traninig , slide window length is 50bp. right now, only use
    #200bp
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


# def crossentropy_cut(y_true,y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
#     mask=K.greater_equal(y_true_f,-0.5)
#     losses = -(y_true_f * K.log(y_pred_f) + (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
#     losses = tf.boolean_mask(losses, mask)
#     masked_loss = tf.reduce_mean(losses)
#     return masked_loss




net=Net()

optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.001,lr=0.001)

track_index_length=int(len(training_index_chip_seq)//batch_size)
print(track_index_length)
for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    for chr in chr_list:

        for track in range(track_index_length):
          input,label=generate_data_for_net(track,batch_size,training_index_chip_seq,chr)
          print(track)

          #input=input.reshape(batch_size,6,length_of_sequence_generate)
          #label=label.reshape(batch_size,1,length_of_sequence_generate)
          #output:100*1*1000
          input=torch.FloatTensor(input.astype(np.float64))
          label=torch.FloatTensor(label.astype(np.int32))
          input=input.permute(2,0,1)
          label=label.permute(2,0,1)
          optimizer.zero_grad()
          outputs = net(input)

          ###log
          #test new loss function
          outputs=torch.clamp(outputs,1e-7, (1. - 1e-7))
          mask=label>=(-0.5)

          loss=-(label*torch.log(outputs)+(1-label)*torch.log(1-outputs))
          loss=torch.masked_select(loss,mask)
          loss=torch.mean(loss)
          loss.backward()
          optimizer.step()
          running_loss=running_loss+loss.item()

    print(epoch,running_loss/len(input))
    test_predict=net(test)

    test_predict=np.matrix(test_predict.detach().numpy()).getA1()
    print('test_predict',np.sum(test_predict))
    print('test_label',np.sum(test_label))
    pos_values, neg_values=score_record(test_label, test_predict, input_digits=None)
    auroc,auprc=calculate_auc(pos_values,neg_values)
    with open('auc_roc_result.txt','a') as f:
        f.write(str(epoch)+'\n'+str(auroc.item())+'\t'+str(auprc.item())+'\n')
    print(auroc,auprc)
    torch.save(net.state_dict(),'./save_model_lr3_wei_3_no_meth_chr_4_20201213.pkl')
    #roc_auc_score_1=roc_auc_score(test_predict.cpu(),test.cpu())



    #print(roc_auc_score_1)
    #print(test_accuracy)

print('Finished Training')
