### all the script

1 whole_process.sh
The script to generated the data with 200bp label, the final data should be 9 columns,

2 FCNN.ipynb
the failure model code which could be run on the google colab GPU

3 SNCNN_without_methylation.py
this script could generate the new model training data and train, save the model without methylation data.

4 SNCNN_with_methylation.py

this script could generate the new model training data and train, save the model with methylation data.

5 generate_test_data.py
generate the test data to be used in the following model evaluation

6 tested_saved_model.py
test the model and plot.

### all the model saved

naming of the model:

save_model_lr3(learning rate)_wei_3(weight decay)_no_meth(with or without methylation)_chr_4(four chorsomes were used)_part(used a part of training data)_20201213.pkl
