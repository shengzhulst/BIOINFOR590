import pyBigWig
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse

#parametmers
cell_lines='K562'
chr_name='chr10'

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
#path3='/data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/final_whole_script/changebedbigwig/'
#methylation_bigwig=pyBigWig.open(f'{path3}{cell_lines}_methylation.bw')

#########open chip-seq as label
chip_seq_TF='CTCF'
path4='/data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/chip_seq_bigwig/'
chip_seq_bigwig=pyBigWig.open(f'{path4}{chip_seq_TF}_{cell_lines}.bigwig')
chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=np.array([249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560])

##################Pickout with signal part
peak_with_signal=pd.read_csv('/data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/chip_seq_bigwig/CTCF_with_peak.bed',sep='\t',header=None)


peak_start=np.random.choice(np.arange(peak_with_signal.shape[0]),2000)
final_matrix_1=np.zeros((1,1))
#calculate how many batch will be generated
final_label_1=np.zeros((1,1))
for index,row in peak_with_signal.loc[peak_start].iterrows():
	empty_output=np.zeros((length_of_sequence_generate,5))
	print(row[0],row[1],row[1]+length_of_sequence_generate)
	final_label_input_training=np.array(chip_seq_bigwig.values(row[0],row[1],row[1]+length_of_sequence_generate))

	for index,the_id in enumerate(list_dna):
		empty_output[:,index]=dict_dna[the_id].values(row[0],row[1],row[1]+length_of_sequence_generate)
	empty_output[:,4]=DNASE_seq_bigwig.values(row[0],row[1],row[1]+length_of_sequence_generate)
	#empty_output[:,5]=methylation_bigwig.values(row[0],row[1],row[1]+length_of_sequence_generate)
	#empty_output[:,5][np.isnan(empty_output[:,5])]=0
	if final_matrix_1.shape[0]==1:
		final_matrix_1=empty_output.T
		final_label_1=final_label_input_training.T
	else:
		final_matrix_1=np.dstack((empty_output.T,final_matrix_1))
		final_label_1=np.dstack((final_label_input_training.T,final_label_1))








################################
chip_seq_start=0
chip_seq_end=num_bp[chr_all.index(chr_name)]
num_sequences_chip_seq=(chip_seq_end-chip_seq_start)//length_of_sequence_generate
temp_bunch_chip_seq=np.arange(num_sequences_chip_seq)[1:5000]


np.random.shuffle(temp_bunch_chip_seq)


final_large_matrix=np.zeros((1,1))
#calculate how many batch will be generated
final_label_matrix=np.zeros((1,1))
count_use=0
for i in list(temp_bunch_chip_seq):
	print(count_use)
	empty_output=np.zeros((length_of_sequence_generate,5))
	start=i*length_of_sequence_generate
	end=start+length_of_sequence_generate
	final_label_input_training=np.array(chip_seq_bigwig.values(chr_name,start,end))


	for index,the_id in enumerate(list_dna):
		empty_output[:,index]=dict_dna[the_id].values(chr_name,start,end)
	empty_output[:,4]=DNASE_seq_bigwig.values(chr_name,start,end)
	# empty_output[:,5]=methylation_bigwig.values(chr_name,start,end)
	# empty_output[:,5][np.isnan(empty_output[:,5])]=0
	if final_large_matrix.shape[0]==1:
		final_large_matrix=empty_output.T
		final_label_matrix=final_label_input_training.T
	else:
		final_large_matrix=np.dstack((empty_output.T,final_large_matrix))
		final_label_matrix=np.dstack((final_label_input_training.T,final_label_matrix))
	count_use+=1


out_put_label=np.dstack((final_label_1,final_label_matrix))
out_put_matrix=np.dstack((final_matrix_1,final_large_matrix))
print(out_put_label.shape)
print(out_put_matrix.shape)
np.save('test_data_K562.npy',out_put_matrix)
np.save('test_label_K562.npy',out_put_label)
