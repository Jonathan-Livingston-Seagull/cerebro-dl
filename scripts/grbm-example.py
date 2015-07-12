from os import path
import sys
import argparse
sys.path.append(path.abspath('../../cerebro-dl'))

import numpy as np
import theano
import theano.tensor as T
import random
import pandas
import cPickle
import gzip

from scipy.io import arff
from sklearn import cross_validation
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from scipy.stats import itemfreq

from cerebro.bm.grbm import GRBM



def run_example(data_path):
    """
    method to demonstrate the usage of grbm.
    :param: data_path path of dataset
    :type: String
    """

    print('... loading data')

    # Load the dataset
    f = gzip.open(data_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    X_train, Y_train = train_set
    print('train x: ', X_train.shape)
    print('train Y: ', Y_train.shape)
    #valid_set_x, valid_set_y = datasets[1]
    X_test, Y_test = test_set
    print('test X: ', X_test.shape)
    print('test Y: ', Y_test.shape)
    print('label count for training data:')
    print(itemfreq(Y_train))
    print('label count for test data:')
    print(itemfreq(Y_test))

    parameters_GRBM = [[200, 2, 10, 0.01, 0.9, 1, 'None',0.1, 0.1, 0.0],
                       [200, 100, 10, 0.01, 0.9, 1, 'L1', 0.2, 0.2, 0.0],
                       [200, 10, 10, 0.01, 0.9, 1, 'L2', 0.3, 0.3, 0.5]]

    for param_grbm in parameters_GRBM:
        grbm = GRBM(random_state=0)
        grbm.n_hidden = param_grbm[0]  
        grbm.grbm_n_iter = param_grbm[1]
        grbm.grbm_batch_size = param_grbm[2]
        grbm.grbm_learning_rate = param_grbm[3]  # fitting time
        grbm.grbm_momentum = param_grbm[4]
        grbm.grbm_n_gibbs_steps = param_grbm[5]
        grbm.penalty = param_grbm[6]
        grbm.C1 = param_grbm[7]
        grbm.C2 = param_grbm[8]
        grbm.pdrop = param_grbm[9]
        
        grbm.fit(X_train, Y_train)

        Y_pred = grbm.predict(X_test)

        score = metrics.accuracy_score(Y_test, Y_pred)

        print('Acc score for test set:', score)
        print("GRBM report:\n%s\n" % (
            metrics.classification_report(
                Y_test,
                Y_pred)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and generates the model for GRBM')
    parser.add_argument('-p','--data_path',required=True,
                        help='path of input data (use mnist data set for easy usage)')
    args = parser.parse_args()
    data_path = args.data_path
    run_example(data_path)