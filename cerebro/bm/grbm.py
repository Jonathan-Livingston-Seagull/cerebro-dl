import theano
import theano.tensor as T
import numpy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

from ..optimizer.gradient_descent import MiniBatchGradientDescent
from cerebro.models import bernoulli_grbm_model


class GRBM(BaseEstimator, ClassifierMixin):
    """ Classification Restricted Boltzmann Machine generative training

        Parameters
        ----------
        n_hidden : integer, optional
            Number of hidden units to use (default: 20)
    
        grbm_n_iter : integer, optional
            Number of epochs to use for training (default: 15)
    
        grbm_learning_rate : float, optional
            Learning rate used during training (default: 0.1)
    
        grbm_batch_size : integer, optional
            Size of a batch used to train the GRBM (default: 10)
            
        grbm_n_gibbs_steps : integer, optional
            Number of Gibbs steps to do in contrastive divergence (default: 1)
            
        penalty: string , optional
            Different types of regularization (default: 'None')
    
        c: float, optional
            Regularization term (default: 0.0) 
        
        pdrop: float, optional 
            probability for dropout regularization (default: 0.0) 
            
        grbm_momentum : float in [0; 1]
            Momentum coefficient (default: 0.9).
    
            The momentum method is a technique for accelerating gradient descent that accumulates a velocity
            vector in directions of persistent reduction in the objective across iterations.
            The update is given by :math:`v_{t+1} = \\alpha v_t - (1 - \\alpha) \\varepsilon \\nabla f(\\theta_t)` and
            :math:`\\theta_{t+1} = \\theta_t + v_{t+1}`.
    
        visible_units : string
            Type of visible units. Can be gaussian(future) or binary.
    
        random_seed : integer, optional
            Random number seed (default: 123)
            
        activation: string, optional.
            type of activation function. (default: 'sigmoid').
    
        noise_type : string, optional
            Type of Additive Noise in layer. Can be 'NOTGaussian' or 'Gaussian' (default: 'NOTGaussian')
    
        noise : float, optional
            Additive Noise in layer. (default: 1.0
    """
    
    def __init__(self, n_hidden=20, visible_units="binary", grbm_learning_rate=0.1, 
                 grbm_momentum=0.9, grbm_batch_size=10, grbm_n_iter=15, grbm_n_gibbs_steps=1,
                 random_state=None,
                 penalty=None, C1 = 0.8, C2=0.8, contraction_level=0.1, activation = 'sigmoid',
                 noise_type ='NOTGaussian',
                 noise = 1.0, pdrop = 0, plot_hidden = False,
                 plot_visible_reconstd = False, plot_reconstd = False):

        self.n_hidden = n_hidden
        self.visible_units = visible_units
        self.grbm_learning_rate = grbm_learning_rate
        self.grbm_momentum = grbm_momentum
        self.grbm_batch_size = grbm_batch_size
        self.grbm_n_iter = grbm_n_iter
        self.grbm_n_gibbs_steps = grbm_n_gibbs_steps
        self.rng = check_random_state(random_state)
        self.plot_hidden = plot_hidden
        self.plot_reconstd = plot_reconstd
        self.plot_visible_reconstd = plot_visible_reconstd
        self.contraction_level = contraction_level
        self.activation = activation
        self.noise_type = noise_type
        self.noise = noise
        self.pdrop = pdrop
        self.C1 = C1
        self.C2 = C2
        self.penalty=penalty
        self.random_state = random_state
        self.grbm_model_ = None
        self.var_x = T.matrix('x')
        self.var_y = T.matrix('Y')

    def fit(self, x, y):
        """Train a restricted Boltzmann machine with Generative approach

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
        # converting one vs all encoding
        label = numpy.ones(shape=(y.shape[0],n_classes))
        for i in range(n_classes):
            one_class = numpy.zeros(n_classes)
            one_class[i] = 1   
            label[y == i] = one_class

        train_set_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)
        train_set_y = theano.shared(numpy.asarray(label, dtype=theano.config.floatX), borrow=True)
        n_train_batches = x.shape[0] // self.grbm_batch_size
        assert (n_train_batches > 0)

        random_state = check_random_state(self.random_state)
        if "binary" == self.visible_units:
            self.grbm_model_ = bernoulli_grbm_model.BernoulliGRBMModel(self.var_x, self.var_y, n_classes, n_features, self.n_hidden, self.rng,
                                                  self.C1, self.C2, self.penalty,self.contraction_level, self.activation,self.noise_type,
                                                  self.noise, self.pdrop)

        else:
            raise ValueError("unknown type of visible units: {0}".format(self.visible_units))

        # initialize storage for the persistent chain (state = hidden layer of chain)
        persistent_chain = theano.shared(numpy.zeros((self.grbm_batch_size, self.n_hidden),
                                                     dtype=theano.config.floatX),
                                         borrow=True)

        grbm_optimizer = MiniBatchGradientDescent(self.grbm_model_, self.grbm_n_iter, self.grbm_learning_rate,
                                                  self.grbm_batch_size, self.grbm_momentum)

        grbm_optimizer.fit(train_set_x, train_set_y, chain_start=persistent_chain, k=self.grbm_n_gibbs_steps)
        
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

        if isinstance(x, theano.compile.SharedVariable):
            test_set_x = x
        else:
            test_set_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)
        pred_prob = self.grbm_model_.p_y_given_x()
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
        
        test_set_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=True)

        y_pred = self.grbm_model_.predict()
        classify = theano.function(inputs=[], outputs=y_pred, givens={self.var_x: test_set_x})
        result = classify()
        return numpy.asarray(result)
