import theano.tensor as T
import theano
import numpy
from theano.tensor.shared_randomstreams import RandomStreams


def ReLU(x):
    """Rectified Linear Unit for activation

    Parameters
    ----------
    x: symbolic variable, shape = [n_samples, n_features]
        input on which to perform ReLU activation

    Returns
    -------
    y : activation of x
    """
    y = T.maximum(0.0, x)
    return y


class BernoulliGRBMModel(object):
    """ Create Classification Restricted Boltzmann Machine for generative training

        :param input_variable: Theano symbolic variable indicating the input matrix(x)
        
        :param input_label: Theano symbolic variable indicating the input matrix(y)

        :param n_visible: Number of visible units
        :type n_visible: integer
        
        :param n_classes: Number of classes
        :type n_classes: integer

        :param n_hidden: Number of hidden units
        :type n_hidden: integer
        
        :param penalty: different types of regularization
        :type penalty: String
        :values: 'l1', 'l2', 'elastic'
        :default: 'None'

        :param C1 : weight for regularization penalty: ['L1']
        :type C1 :float, optional
            
        :param C2 : weight for regularization penalty: ['L2']
        :type: C2 :float, optional

        :param pdrop: probability for dropout regularization
        :type pdrop: float
        :default: 0.0 
        
        :param rng: Random number generation
        :type rng: :class:`numpy.random.RandomState`
    """
        
    def __init__(self, input_variable, input_label, n_classes, n_visible, n_hidden, rng,
                 C1=0.8, C2=0.8, penalty=None, contraction_level=0.1, activation = 'sigmoid',
                 noise_type = 'None',
                 noise = 1.0, pdrop = 0):

        self.input_x = input_variable
        self.input_y_binary = input_label
        self.n_classes = n_classes
        self.binary_label = theano.shared(self.binarise_label(), name='binary_label', borrow=True)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = theano.shared(numpy.asarray(rng.normal(0, 0.01, size=(n_visible, n_hidden)), dtype=theano.config.floatX),
                                      name="W")
        self.U = theano.shared(numpy.asarray(rng.normal(0, 0.01, size=(n_classes, n_hidden)), dtype=theano.config.floatX),
                              name="U")
        self.hbias = theano.shared(value=numpy.zeros(n_hidden,
                                                    dtype=theano.config.floatX),
                                  name='hbias', borrow=True)
        self.vbias = theano.shared(value=numpy.zeros(n_visible,
                                                    dtype=theano.config.floatX),
                                  name='vbias', borrow=True)
        self.lbias = theano.shared(value=numpy.zeros(n_classes,
                                                    dtype=theano.config.floatX),
                                  name='lbias', borrow=True)
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))

        self.params = [self.W, self.U, self.hbias, self.lbias]

        self.penalty = penalty
        self.contraction_level = contraction_level
        self.activation = activation
        self.noise_type = noise_type
        self.noise = noise
        #Penalty parameter C1 of the error term for penalty L1
        self.C1 = C1
        #Penalty parameter C1 of the error term for penalty L2
        self.C2 = C2
        #Penalty parameter l1_ratio and alpha for penalty 'elastic-net'
        self.l1_ratio = C1/(C1+C2)
        self.alpha = C1 + C2
        self.pdrop = pdrop

    def p_y_given_x(self):
        """
        predict the probabilities of classes for given input
        
        :math: `p(y=e_c | x) = \\softmax(-Free energy)`
        
        Returns
        -------
        p_y_given_x : symbolic, shape = [n_samples, n_classes]
                    Returns the probability of the sample for each class in the model.
        """
        
        free_energy = self._free_energy(self.input_x)
        prob =  T.nnet.softmax(-free_energy)
        return prob

    def predict(self):
        """
        predict the output classes for given input
        
        Returns
        -------
        c : array-like, shape = [n_samples]
            Predicted class labels per sample
        """
        
        return T.argmax(self.p_y_given_x(), axis=1)

    def _free_energy(self, v):
        """
        Computes the free energy 
        
        :math: `F(x,y) = -d.T * y - \\sum(\\softplus(c_j + W_j * x + U_j * y ))`
        Parameters
        ----------
        v : symbolic, shape = [n_samples, n_features]
        
        Returns
        -------
        Free energy : symbolic, Free energy matrix with shape of n_samples * n_classes
            Input to predict probability
        """
        
        reshape_term1 = T.dot(v, self.W)
        reshape_term1 = T.reshape(reshape_term1, (T.as_tensor_variable(reshape_term1.shape[0]),T.as_tensor_variable(reshape_term1.shape[1]),
                                                  T.as_tensor_variable(1)), ndim=3)
        reshape_term2 = T.reshape(self.hbias,(T.as_tensor_variable(1),T.as_tensor_variable(self.hbias.shape[0])
                                               , T.as_tensor_variable(1)), ndim=3)
        return (- T.dot(self.lbias, self.binary_label)
                - T.nnet.softplus(reshape_term1
                               + T.dot(self.U.T, self.binary_label)
                               + reshape_term2).sum(axis=1))

    def binarise_label(self):
        """
        create one hot representation of labels
        
        Example: if n_classes = 2 this method generates [[1 0], [0 1]]
        """
        
        label = numpy.zeros(shape=(self.n_classes,self.n_classes))
        for i in range(self.n_classes):
            label[i][i] = 1
        return label

    def parameters(self):
        return self.W, self.U, self.hbias, self.lbias

    def get_input(self):
        return self.input_x, self.input_y_binary

    def predict_function(self):
        return T.argmax(self.p_y_given_x, axis=1)
    
    def drop_output(self, input, pdrop=0.5):
        """ dropout some units with given probability
        
        Parameters
        ----------
        input : layer or weight matrix on which dropout  is applied, symbolic, shape = [n_samples, n_hidden]

        pdrop : float  between 0. and 1. 

        Returns
        -------
        dropped_output : symbolic variable, shape = [n_samples, n_hidden]
        """
                
        shrd_rnd_strm = self.theano_rng
        mask = shrd_rnd_strm.binomial(n=1, p=1 - pdrop, size=input.shape)
        dropped_output = input * T.cast(mask, dtype = theano.config.floatX)
        return dropped_output

    def propagate_visible(self, vis_x, vis_y):
        """Propagate visible units activation to hidden units.

        :math:`P(h_j = 1 | x, y) = sigmoid(c_i + W_j * x +  U_j*y)`

        Parameters
        ----------
        vis_x : symbolic vector, shape = [1, n_visible]
            visible sample
        vis_y : symbolic vector, shape = [1, n_classes]

        Returns
        -------
        pre_sigmoid_activation : symbolic function
            Input to activation function
        output : symbolic function
            Output of activation function
        """
        
        pre_sigmoid_activation = self.hbias + T.dot(vis_x, self.W) +  T.dot(vis_y, self.U)
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def propagate_hidden(self, hidden):
        """Propagate hidden units activation to visible units

        :math:`P(x_i = 1 | h) = sigmoid(b_i + h.T * W_i)` 
        :math:`P(y = e_c | h) = softmax( d.T * y + h.T * U * y ) `

        Parameters
        ----------
        hidden : symbolic vector, shape = [1, n_hidden]
            hidden sample

        Returns
        -------
        pre_sigmoid_activation : symbolic function
            Input to activation function
        output : symbolic function
            Output of activation function
        """
        
        pre_sigmoid_activation = self.vbias + T.dot( hidden, self.W.T)
        vy1_mean = T.nnet.softmax((T.dot(self.lbias, self.binary_label)+ T.dot(T.dot(hidden ,self.U.T),self.binary_label)))
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation), vy1_mean]

    def sample_hidden(self, vis_sample_x, vis_sample_y):
        """Returns state of hidden units, given the visible units

        Parameters
        ----------
        vis_sample : symbolic vector, shape = [1, n_visible]
            visible sample
        """
        
        pre_sigmoid_hidden, hidden = self.propagate_visible(vis_sample_x, vis_sample_y)
        hidden_sample = self.theano_rng.binomial(size=hidden.shape,
                                             n=1, p=hidden,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_hidden, hidden, hidden_sample]

    def sample_visible(self, hidden_sample):
        """Returns state of visible units, given the hidden units

        Parameters
        ----------
        hidden_sample : symbolic vector, shape = [1, n_hidden]
            hidden sample
        """
        
        pre_sigmoid_vis_x, vis_x, vis_y = self.propagate_hidden(hidden_sample)
        vis_x_sample = self.theano_rng.binomial(size=vis_x.shape,
                                             n=1, p=vis_x,
                                             dtype=theano.config.floatX)
        vis_y_sample = self.theano_rng.binomial(size=vis_y.shape,
                                             n=1, p=vis_y,
                                           dtype=theano.config.floatX)
        return [pre_sigmoid_vis_x, vis_x, vis_x_sample, vis_y, vis_y_sample]

    def gibbs_from_hidden(self, hidden_sample):
        """Do one step of Gibbs sampling, starting from hidden units

        :math:`v^{(n+1)} = sigmoid(W \\cdot h^{(n+1)} + b_h)`
        
        Parameters
        ----------
        :param hidden_sample: symbolic, shape=[n_samples, n_hidden]
        """
        
        pre_sigmoid_vis_x, vis_x, vis_x_sample, vis_y, vis_y_sample = self.sample_visible(hidden_sample)
        pre_sigmoid_hidden, new_hidden, new_hidden_sample = self.sample_hidden(vis_x_sample, vis_y_sample)
        return [pre_sigmoid_vis_x, vis_x, vis_x_sample, vis_y, vis_y_sample,
                pre_sigmoid_hidden, new_hidden, new_hidden_sample]

    def gibbs_from_observed(self, vis_sample_x, vis_sample_y):
        """Do one step of Gibbs sampling, starting from visible units

        :math:`h^{(n+1)} = sigmoid(W^T \\cdot v^{(n)} + b_v)`
        
        Parameters
        ----------
        :param vis_sample_x: symbolic, shape=[n_samples, n_features]
        :param vis_sample_y: symbolic, shape=[n_samples, n_classes]
        """
                
        pre_sigmoid_hidden, hidden, hidden_sample= self.sample_hidden(vis_sample_x, vis_sample_y)
        pre_sigmoid_vis_x, vis_x, vis_x_sample, vis_y, vis_y_sample = self.sample_visible(hidden_sample)
        return [pre_sigmoid_hidden, hidden, hidden_sample,
                pre_sigmoid_vis_x, vis_x, vis_x_sample, vis_y, vis_y_sample]

    def energy_function(self, observed_x, observed_y):
        """
        Function to compute the free energy 
         
        :math: `F(x,y) = -d.T * y - \\sum(\\softplus(c_j + W_j * x + U_j * y ))`
        Parameters
        ----------
        observed_x : symbolic, input variable matrix, shape = [n_samples, n_features]
        observed_y : symbolic, input label matrix, [n_samples, n_classes]
        Returns
        -------
        Free energy : Free energy matrix with shape of n_samples * n_classes, symbolic
        """
        
        obias_term = T.dot(observed_y, self.lbias)
        hidden_term = T.sum(T.nnet.softplus((self.hbias + T.dot( observed_x, self.W) + T.dot(observed_y, self.U) )), axis=1)
        return -obias_term - hidden_term


    def get_jacobian(self, hidden, activation='sigmoid'):
        """Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis
        Parameters
        ----------
        hidden: symbolic variable
            hidden layer units

        activation:  string, optional.
                    type of activation function. (default: 'sigmoid').
        Returns
        -------
        contractive_cost : symbolic, jacobian of the hidden layer
        """
        
        if activation == 'sigmoid':
            contractive_cost = T.mean(((hidden * (1 - hidden))**2) * T.sum(self.W**2, axis=0))
        elif activation == 'softplus':
            contractive_cost = T.mean( hidden * T.sum(self.W**2, axis=0))
        return contractive_cost


    def get_hidden_values(self, x, activation='sigmoid'):
        """ Computes the values of the hidden layer
        Parameters
        ----------
        x: symbolic variable , shape = [n_samples, n_features]
.
        Returns
        -------
        hidden unit: symbolic, [n_samples, n_hidden]
         """
        
        if activation == 'sigmoid':
            return T.nnet.sigmoid(T.dot(x, self.W) + self.hbias)
        elif activation == 'softplus':
            return T.nnet.softplus(T.dot(x, self.W) + self.hbias)
    
    def grbm_cost(self, chain_start=None, k=1):
        """calculate cost for grbm

        Parameters
        ----------
        chain_start : shared variable, shape (batch_size, n_hidden)
            Old state of the Gibbs chain
        k : integer
            number of Gibbs step to do

        Returns
        -------
        cost : float
            Pseudo-likelihood

        updates : dict
            Update rules for weights and biases as well as an update to shared variable chain_chart
        """
        
        if chain_start is None:
            pre_sigmoid_ph, ph_mean, ph_sample = self.sample_hidden(self.input_x, self.input_y_binary)
            chain_start = ph_sample

        [pre_sigmoid_nvs_x, nv_x_means, nv_x_samples, nv_y_means, nv_y_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_from_hidden, outputs_info=[None, None, None, None, None, None, None, chain_start], n_steps=k)

        chain_end_x = nv_x_samples[-1]
        chain_end_y = nv_y_samples[-1]

        cost = T.mean(self.energy_function(self.input_x, self.input_y_binary)) - T.mean(self.energy_function(chain_end_x, chain_end_y))

        # different regularization will be moved to common class in future
        if self.penalty == 'L2':
            cost += self.C2 * 0.5 * T.sum(self.W ** 2)
        elif self.penalty == 'JACOBIAN':
            h = self.get_hidden_values(self.input_x, self.activation)

            if self.noise_type == 'Gaussian':
                h += self.theano_rng.normal(size=self.hbias.shape, avg=0.0, std=theano.tensor.sqrt(self.noise))
            contractive_cost = self.get_jacobian(h, self.activation)
            # add contractive regularizer
            cost += self.contraction_level * contractive_cost

        elif self.penalty == 'L1':
            cost += self.C1 * 0.5 * T.sum(abs(self.W))
        elif self.penalty == 'elastic-net':
            cost +=  self.alpha * self.l1_ratio * T.sum(abs(self.W))  \
                    + 0.5 * self.alpha * (1 - self.l1_ratio) * T.sum(self.W ** 2)
                    
        return cost, updates, chain_start, nh_samples, chain_end_x, chain_end_y        

    def parameters_gradient_updates(self, chain_start=None, k=1):
        """Do one step of persistent contrastive divergence (PCD)

        Parameters
        ----------
        chain_start : shared variable, shape (batch_size, n_hidden)
            Old state of the Gibbs chain
        k : integer
            number of Gibbs step to do

        Returns
        -------
        cost : float
            Pseudo-likelihood

        updates : dict
            Update rules for weights and biases as well as an update to shared variable chain_chart
        """
        
        cost, updates, chain_start, nh_samples, chain_end_x, chain_end_y = self.grbm_cost(chain_start, k)
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end_x, chain_end_y])

        # Note that this works only if chain_start is a shared variable
        if chain_start is not None:
            if self.pdrop:
                updates[chain_start] = self.drop_output(nh_samples[-1], self.pdrop)
            else:
                updates[chain_start] = nh_samples[-1]

        return gparams, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        bit_i_idy = theano.shared(value=0, name='bit_i_idy')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input_x)
        yi = self.input_y_binary

        # calculate free energy for the given bit configuration
        fe_xi = self.energy_function(xi,yi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        yi_flip = T.set_subtensor(yi[:, bit_i_idy], 1 - yi[:, bit_i_idy])

        # calculate free energy with bit flipped
        fe_flip = self.energy_function(xi_flip,yi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean((self.n_visible + self.n_classes) * T.log(T.nnet.sigmoid(fe_xi -
                                                            fe_flip)))

        if self.penalty == 'L2':
            cost += self.C2 * 0.5 * T.sum(self.W ** 2)
        elif self.penalty == 'JACOBIAN':
            h = self.get_hidden_values(self.input_x, self.activation)
            if self.pdrop:
                h = self.drop_output(h, self.pdrop)

            if self.noise_type == 'Gaussian':
                h += self.theano_rng.normal(size=self.hbias.shape, avg=0.0, std=theano.tensor.sqrt(self.noise))
            contractive_cost = self.get_jacobian(h, self.activation)
        
            # add contractive regularizer
            cost += self.contraction_level * contractive_cost

        elif self.penalty == 'L1':
            cost += self.C1 * 0.5 * T.sum(abs(self.W))
        elif self.penalty == 'elastic-net':
            cost +=  self.alpha * self.l1_ratio * T.sum(abs(self.W))  \
                    + 0.5 * self.alpha * (1 - self.l1_ratio) * T.sum(self.W ** 2)

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        updates[bit_i_idy] = (bit_i_idy + 1) % self.n_classes

        return cost

    def get_reconstruction_cost(self):
        """Compute the mean-squared reconstruction error

        It does one Gibbs step starting from the training data and compares the obtained sample with the training data.
        The reconstruction error is calculated deterministically, i.e. no sampling is done.
        """
        pre_sigmoid_hidden, hidden, hidden_sample, pre_sigmoid_vis_x, \
        new_vis_x, new_vis_x_sample, new_vis_y, new_vis_y_sample = self.gibbs_from_observed(self.input_x, self.input_y_binary)
        x_cost_mean = ((self.input_x - new_vis_x) ** 2).sum(axis=1).mean()
        y_cost_mean = ((self.input_y_binary - new_vis_y) ** 2).sum(axis=1).mean()
        return x_cost_mean + y_cost_mean

    def get_monitoring_cost(self, updates):
        # pseudo-likelihood is a better proxy for PCD
        monitoring_cost = self.get_pseudo_likelihood_cost(updates)

        return monitoring_cost

    def get_validation_error(self):
        """Return the validation error on validation set

        Returns
        -------
        validation error : symbolic variable
        """
        
        pre_sigmoid_hidden, hidden, hidden_sample, pre_sigmoid_vis_x, new_vis_x, \
        new_vis_x_sample, new_vis_y, new_vis_y_sample = self.gibbs_from_observed(self.input_x, self.input_y_binary)
        x_cost_mean = ((self.input_x - new_vis_x) ** 2).sum(axis=1).mean()
        y_cost_mean = ((self.input_y_binary - new_vis_y) ** 2).sum(axis=1).mean()
        return x_cost_mean + y_cost_mean
