chr=$1
TF_name=$2
column_for_cell_line=$3
cell_name=$4
####the start part is to deal with the label

cd /data/shiting/bioinfor590_deep_learning/training_labels
#gzip -d $2.train.labels.tsv.gz &&
#sed '1d' $2.train.labels.tsv >study_label_part/$2.train.labels.bed &&

cd study_label_part
#bedextract $1 $2.train.labels.bed >$2.train.labels_$1.bed

#cut -f1,2,3,$3 $2.train.labels_$1.bed >/data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/store_label/$2_$1_$4_label.bed

#######after the label part is the methylation part
####### we default thought the methylation data has been generated
cd /data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/methylation_dealing
if [ -f $4_$1.bed ];then
	echo "methylation data is avaiable"
else
	echo "generate methylation data"
	bedextract $1 /data/shiting/methy_enrich_main_code/covert_hg19_to_hg38/DNA_methylation_back_to_hg19/$4/*_hg19.bed>$4_$1.bed
	cut -f1,2,3,4,11 $4_$1.bed >./five_column/$4_$1_five.bed
fi

bedmap --echo --mean --stdev /data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/store_label/$2_$1_$4_label.bed ./five_column/$4_$1_five.bed >/data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/mid_data/$4_methy_$1_$2.bed

##################after the methylation part is the DNASE part
cd /data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/Dnase_seq_dealing

if [ -f DNASE_$1_$4.bed ];then
        echo "DNASE-seq data is avaiable"
else
        echo "generate DNASE data"
        /data/shiting/bioinfor590_deep_learning/DNASE/fold_coverage_wiggles/bigwigtobedgraph/./bigWigToBedGraph -chrom=$1  /data/shiting/bioinfor590_deep_learning/DNASE/fold_coverage_wiggles/DNASE.$4.fc.signal.bigwig ./DNASE_$1_$4.bed
	awk '{print $1,"\t",$2,"\t",$3,"\t",".","\t",$4}' ./DNASE_$1_$4.bed > ./mature_data/$1_$4.bed
fi


bedmap --echo --mean --stdev /data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/mid_data/$4_methy_$1_$2.bed ./mature_data/$1_$4.bed >/data/shiting/bioinfor590_deep_learning/data_dealing_whole_process/mid_data/$4_mid2_$1_$2.bed








