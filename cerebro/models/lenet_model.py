import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams
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
    """
    Lenet model to construct deep layers of network. Filter size is used 5 * 5. In future it will be 
    handled as a parameter. in case if you want to change the filter size make sure you adujust other
    parameters accordingly
    """
    
    def __init__(self, var_input, var_label, nkerns, batch_size, n_features, n_classes, rng):
        self.var_input = var_input
        self.var_label = var_label
        self.rng = rng
        self.nkerns = nkerns
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_classes = n_classes
        #be sure that input is perfect square
        input_shape = int(math.sqrt(n_features))
        self.layer0_input = self.var_input.reshape((self.batch_size, 1, input_shape, input_shape))
    
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        self.layer0 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer0_input,
            image_shape=(self.batch_size, 1, input_shape, input_shape),
            filter_shape=(self.nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )
        #self.layer0_output = self.layer0.output.eval({x:train_set_x.get_value()[:batch_size]})
        #print(self.layer0_output.shape)
        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        input_shape = int ((input_shape - 5 + 1) / 2)
        self.layer1 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer0.output,
            image_shape=(self.batch_size, self.nkerns[0], input_shape, input_shape),
            filter_shape=(self.nkerns[1], self.nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
        
        input_shape = int ((input_shape - 5 + 1) / 2)
        #layer1_output = layer1.output.eval({x:train_set_x.get_value()[:batch_size]})
        #print(layer1_output.shape)
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        self.layer2_input = self.layer1.output.flatten(2)
        #print(layer1_output.flatten(2).shape)
        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayerModel(
            self.rng,
            input=self.layer2_input,
            n_in=self.nkerns[1] * input_shape * input_shape,
            n_out=500,
            activation=T.tanh
        )
    
        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegressionModel(self.layer2.output,self.var_label, 500, self.n_classes)
        
    def parameters(self):
        return self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params

    def predict_function(self):
        return T.argmax(self.layer3.p_y_given_x, axis=1)
    
    def predict_prob(self):
        return self.layer3.p_y_given_x
    
    def get_monitoring_cost(self, updates=None):
        cost = self.layer3.get_monitoring_cost(updates)
        return cost

    def parameters_gradient_updates(self):
        cost = self.get_monitoring_cost()
        return T.grad(cost, self.parameters()), None

    def get_input(self):
        return self.var_input, self.var_label

    def get_validation_error(self):
        return T.mean(T.neq(self.predict_function(), self.var_label))

