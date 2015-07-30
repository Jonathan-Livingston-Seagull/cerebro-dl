import theano
import theano.tensor as T
import numpy

from sklearn.base import BaseEstimator, ClassifierMixin

from ..optimizer.gradient_descent import MiniBatchGradientDescent
from cerebro.models.logistic_regression_model import LogisticRegressionModel


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Linear logistic regression model

    Parameters
    ----------
    learning_rate : float
        The learning rate. Defaults to 6
        The learning rate is normalized with respect to the batch_size as learning_rate / batch_size

    batch_size : integer
        Number of samples to use for one iteration. Defaults to 600.

    n_iter : integer
        The number of gradient updates (aka epochs). Defaults to 1000.
    """

    def __init__(self, learning_rate=6, batch_size=600, n_iter=1000):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        self.model_ = None
        self.var_x = T.matrix('x')
        self.var_y = T.ivector('y')

    def fit(self, x, y):
        """Fit model with batch stochastic gradient descent

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples]
            Class labels of training samples
        Returns
        -------
        self : an instance of self
        """

        n_features = x.shape[1]
        n_classes = len(numpy.unique(y))

        train_set_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)
        train_set_y = theano.shared(numpy.asarray(y, dtype=numpy.int32), borrow=True)

        self.model_ = LogisticRegressionModel(self.var_x, self.var_y, n_features, n_classes)

        optimizer = MiniBatchGradientDescent(self.model_, self.n_iter, self.learning_rate, self.batch_size, 0)
        optimizer.fit(train_set_x, train_set_y)

        return self

    def predict_proba(self, x):
        """Probability estimates for each class.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Samples of which class should be predicted

        Returns
        -------
        p : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model.
        """

        n_features = self.model_.shape[0]
        if x.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (x.shape[1], n_features))

        test_set_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)

        pred_prob = self.model_.p_y_given_x
        classify = theano.function(inputs=[], outputs=pred_prob, givens={self.var_x: test_set_x})
        result = classify()
        return result

    def predict(self, x):
        """Predict class labels of samples in x

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Samples of which class should be predicted

        Returns
        -------
        c : array-like, shape = [n_samples]
            Predicted class labels per sample
        """

        n_features = self.model_.shape[0]
        if x.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (x.shape[1], n_features))

        test_set_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)

        y_pred = self.model_.predict_function()
        classify = theano.function(inputs=[], outputs=y_pred, givens={self.var_x: test_set_x})
        result = classify()
        return numpy.asarray(result)
