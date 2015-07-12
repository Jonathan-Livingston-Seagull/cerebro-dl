import theano
import theano.tensor as T
import numpy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

from ..optimizer.gradient_descent import MiniBatchGradientDescent
from cerebro.models import lenet_model

class LeNet(BaseEstimator, ClassifierMixin):
    """Lenet classifier

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

    def __init__(self, learning_rate=0.1, momentum=0.9, batch_size=500, n_epochs=100, nkerns=[20, 50], random_state=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.nkerns = nkerns
        self.rng = check_random_state(random_state)
        self.model_ = None
        self.var_x = T.matrix('x')
        self.var_y = T.ivector('y')

    def fit(self, x, y, validation_x=None, validation_y=None):
        """Fit model with batch stochastic gradient descent

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples]
            Class labels of training samples

        validation_x : array-like
            Validation data set

        validation_y : array_like
            Corresponding class labels of validation set

        Returns
        -------
        self : an instance of self
        """

        self.n_features = x.shape[1]
        self.n_classes = len(numpy.unique(y))
        self.model_ = lenet_model.LeNetModel(self.var_x, self.var_y, self.nkerns, self.batch_size, self.n_features, self.n_classes, self.rng)

        if validation_x is None or validation_y is None:
            stopping_criteria = None
            
        train_set_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)
        train_set_y = theano.shared(numpy.asarray(y, dtype=numpy.int32), borrow=True)
        optimizer = MiniBatchGradientDescent(self.model_, self.n_epochs, self.learning_rate, self.batch_size, self.momentum)
        optimizer.fit(train_set_x, train_set_y, validation_x=validation_x, validation_y=validation_y)

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

        n_samples = x.shape[0]
        n_train_batches = n_samples // self.batch_size
        test_set_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)
        var_index = T.lscalar()
        givens = {self.var_x: test_set_x[var_index * self.batch_size:(var_index + 1) * self.batch_size]}
        pred_prob = self.model_.predict_prob()
        classify = theano.function([var_index], outputs=pred_prob, givens=givens, name="predict_probability")
        result = []
        for batch_index in range(n_train_batches):
            result.append(classify(batch_index))
        return numpy.reshape(result, newshape=(n_samples, self.n_classes))
       

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

        n_samples = x.shape[0]
        n_train_batches = n_samples // self.batch_size
        test_set_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)
        var_index = T.lscalar()
        givens = {self.var_x: test_set_x[var_index * self.batch_size:(var_index + 1) * self.batch_size]}
        pred_prob = self.model_.predict_function()
        classify = theano.function([var_index], outputs=pred_prob, givens=givens, name="predict_probability")
        result = []
        for batch_index in range(n_train_batches):
            result.append(classify(batch_index))
        return numpy.reshape(result, newshape=(n_samples, ))
