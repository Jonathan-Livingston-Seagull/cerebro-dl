from os import path
import sys
import argparse
sys.path.append(path.abspath('../../cerebro-dl'))


import cPickle
import gzip

from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize

from cerebro.convnets.lenet import LeNet



def run_example(data_path):
    """
    method to demonstrate the usage of lenet.
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
    
    #grid search parameters
    parameters_convnet = [{'batch_size':100, 'learning_rate':0.1, 'nkerns':[20, 50], 'n_epochs':10}]

    for param_conv in parameters_convnet:
        lenet = LeNet(random_state=0)
        lenet.batch_size = param_conv['batch_size']
        lenet.learning_rate = param_conv['learning_rate']
        lenet.nkerns = param_conv['nkerns']
        lenet.n_epochs = param_conv['n_epochs']
        X_train = X_train[0:1000,:]
        Y_train = Y_train[0:1000]
        X_test = X_test[0:1000,:]
        Y_test = Y_test[0:1000]
        lenet.fit(X_train, Y_train)

        y_prob = lenet.predict_proba(X_test)
        print(len(y_prob))
        print(y_prob.shape)
        y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        #print('probas',y_prob)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        print('roc area is ', roc_auc["micro"])
        
        Y_pred = lenet.predict(X_test)

        score = accuracy_score(Y_test, Y_pred)
        print("accuracy score is: ", score)

        print("GRBM report:\n%s\n" % (
        classification_report(
            Y_test,
            Y_pred)))
        
        #saving the model, change the model path as needed, it helps in incremental training
        #print('saving the model to the disk')
        #joblib.dump(lenet, 'lenet.pkl')
        #print('model is saved')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and generates the model for GRBM')
    parser.add_argument('-p','--data_path',required=True,
                        help='path of input data (use mnist data set for easy usage)')
    args = parser.parse_args()
    data_path = args.data_path
    run_example(data_path)
    
