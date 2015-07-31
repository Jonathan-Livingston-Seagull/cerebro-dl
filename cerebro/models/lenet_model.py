import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import numpy
import math

from .hidden_layer_model import HiddenLayerModel
from .logistic_regression_model import LogisticRegressionModel


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

class LeNetModel:
    """ Lenet model to construct deep layers of network.

        :param input_variable: Theano symbolic variable indicating the input matrix(x)

        :param input_label: Theano symbolic variable indicating the input matrix(y)

        :param nkerns: Each element represents how much depth in a single layer
        :type nkerns: list ( the no of layers will be automatically calculated from the size of the list )

        :param _n_classes: Number of classes
        :type _n_classes: integer

        :param filter_sizes: Each element represents size of the filter in a layer
        :type filter_sizes: list

        :param pool_sizes: Each element represents size of the pooling in a layer
        :type pool_sizes: list

        :param fully_connected_n_output: number of hidden units in fully connected layer
        :type fully_connected_n_output: integer

    """
    
    def __init__(self, input_variable, input_label, nkerns, batch_size, n_features, n_classes, filter_sizes, pool_sizes, fully_connected_n_output):
        self.input_variable = input_variable
        self.input_label = input_label
        self.rng = numpy.random.RandomState(23455)
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_classes = n_classes
        self.params = []
        self.conv_pool_layers = []
        self.n_conv_pool_layers = len(nkerns)
        input_shape = int(math.sqrt(n_features))
        # create n layers ( n = size of kernels) be sure you checked the filter sizes
        # matches with input size ( look at the formulae in literature)
        for i in xrange(self.n_conv_pool_layers):
            if i == 0:
                #be sure that input is perfect square
                input_dimension = 1
                layer_input = self.input_variable.reshape((self.batch_size, 1, input_shape, input_shape))
            else:
                input_shape = (input_shape - filter_sizes[i-1] + 1) / 2
                input_dimension = nkerns[i-1]
                layer_input = self.conv_pool_layers[-1].output

            lenet_conv_pool_layer = LeNetConvPoolLayer(
            self.rng,
            input=layer_input,
            image_shape=(self.batch_size, input_dimension, input_shape, input_shape),
            filter_shape=(nkerns[i], input_dimension, filter_sizes[i], filter_sizes[i]),
            poolsize=(pool_sizes[i], pool_sizes[i])
            )

            self.conv_pool_layers.append(lenet_conv_pool_layer)
            self.params.extend(lenet_conv_pool_layer.params)

        input_shape = (input_shape - filter_sizes[-1] + 1) / 2
        layer_input = self.conv_pool_layers[-1].output.flatten(2)

        # construct a fully-connected sigmoidal layer
        fully_connected_layer = HiddenLayerModel(
            self.rng,
            input=layer_input,
            n_in=nkerns[-1] * input_shape * input_shape,
            n_out=fully_connected_n_output,
            activation=T.tanh
        )

        self.conv_pool_layers.append(fully_connected_layer)
        self.params.extend(fully_connected_layer.params)

        # Final logistic regression classification layer
        self.logistic_regression_layer = LogisticRegressionModel(
            self.conv_pool_layers[-1].output, self.input_label, fully_connected_n_output, self.n_classes
        )
        self.params.extend(self.logistic_regression_layer.params)

    def parameters(self):
        return self.params

    def predict_function(self):
        return T.argmax(self.logistic_regression_layer.p_y_given_x, axis=1)

    def predict_prob(self):
        return self.logistic_regression_layer.p_y_given_x

    def get_monitoring_cost(self, updates=None):
        cost = self.logistic_regression_layer.get_monitoring_cost(updates)
        return cost

    def parameters_gradient_updates(self):
        cost = self.get_monitoring_cost()
        return T.grad(cost, self.parameters()), None

    def get_input(self):
        return self.input_variable, self.input_label

    def get_validation_error(self):
        return T.mean(T.neq(self.predict_function(), self.var_label))