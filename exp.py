"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

from utils import preprocess_data, split_train_dev_test,read_digits,predict_and_eval,make_param_combinations,tune_hparams

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
C_ranges = [0.1,1,2,5,10]
param_list_dict = {'gamma':gamma_ranges,'C':C_ranges}

test_size_ranges = [0.1, 0.2, 0.3]
dev_size_ranges = [0.1, 0.2, 0.3]
split_size_list_dict = {'test_size':test_size_ranges,'dev_size':dev_size_ranges}
                        
x,y = read_digits()
splits = make_param_combinations(split_size_list_dict)

for split in splits:

    # Data splitting: Split data into train, test and dev as per given test and dev sizes
    X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(x,y, **split)

    # Data preprocessing
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    X_dev = preprocess_data(X_dev)

    best_hparams,best_model, best_accuracy =  tune_hparams(X_train,y_train,X_dev,y_dev,param_list_dict)
    
    train_acc,predicted = predict_and_eval(best_model,X_train,y_train)
    test_acc,predicted = predict_and_eval(best_model,X_test,y_test)
    dev_acc,predicted = predict_and_eval(best_model,X_dev,y_dev)

    print("Test size=%g, Dev size=%g, Train_size=%g, Train_acc=%g, Test_acc=%g, Dev_acc=%g" % (split['test_size'],split['dev_size'],1-split['test_size']-split['dev_size'],train_acc,test_acc,dev_acc) ,sep='')
    print("best hparams=",dict([(x,best_hparams[x]) for x in param_list_dict.keys()]))