import numpy
import signal
import theano
import theano.tensor as T
from collections import OrderedDict

class MiniBatchGradientDescent:
    """Mini Batch Gradient Descent Optimization

    A single gradient update step looks like
    `update = parameter + previous_update * momentum - gradient * learning_rate / batch_size`

    Parameters
    ----------
    model : instance
        An instance of the model that is to be trained.

        A model must provide the following methods: `get_input`, `parameters`, `parameters_gradient_updates`, and
        `get_monitoring_cost`.

           * `get_input` returns a tuple of symbolic input variables (usually a feature matrix and label vector).
           * `parameters` returns a tuple of parameters of the model (e.g. weights and biases).
           * `parameters_gradient_updates`: return a tuple of gradients (wrt parameters, and optionally a OrderedDict
           of updates or None. The order of the gradients must be the same as in `parameters`.
           Keyword arguments **kwargs are passed to this function.
           * `get_monitoring_cost`: A cost function that monitors the progress and is printed after each iteration.

    n_iter : integer
        Maximum number of iterations (aka epochs) over the training data

    learning_rate : float
        The learning rate. The learning rate is normalized by the batch_size as `learning_rate/batch_size`

    batch_size : integer, positive
        Number of samples to consider in one gradient update step

    momentum : float, [0; 1]
        Defines trade-off between previous update step and current update step

    stopping_criteria : instance
        Instance of class that implements a stopping criteria on a separate validation set or `None` to disable
    """

    def __init__(self, model, n_iter, learning_rate, batch_size, momentum, stopping_criteria=None):
        self.model = model
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.stopping_criteria = stopping_criteria
        self._do_stop = False

    @staticmethod
    def _make_shared(array, dtype):
        if isinstance(array, theano.compile.SharedVariable):
            return array
        else:
            return theano.shared(numpy.asarray(array, dtype=dtype), borrow=True)

    def _signal_handler(self, signum, frame):
        print("Caught signal {0}".format(signum))
        if signum == signal.SIGTERM:
            self._do_stop = True

    def fit(self, X, y=None, validation_x=None, validation_y=None, bin_y = None, **kwargs):
        assert (self.learning_rate > 0)
        assert (0 <= self.momentum <= 1)

        # Convert input to shared variables, if necessary
        if isinstance(X, theano.compile.SharedVariable):
            train_set_x = X
            n_samples = X.get_value(borrow=True).shape[0]
        else:
            train_set_x = theano.shared(numpy.asarray(X, dtype=theano.config.floatX), borrow=True)
            n_samples = X.shape[0]

        if y is not None:
            if isinstance(y, theano.compile.SharedVariable):
                train_set_y = y
            else:
                train_set_y = self._make_shared(y, theano.config.floatX)
            

        if bin_y is not None:
            if isinstance(bin_y, theano.compile.SharedVariable):
                train_set_y_binary = bin_y
            else:
                train_set_y_binary = self._make_shared(bin_y, theano.config.floatX)

        if validation_x is not None:
            valid_set_x = self._make_shared(validation_x, theano.config.floatX)
        if validation_y is not None:
            valid_set_y = self._make_shared(validation_y, theano.config.floatX)

        var_index = T.lscalar()  # index to a [mini]batch
        n_train_batches = n_samples // self.batch_size

        assert (n_train_batches > 0)

        # retrieve the symbolic variables denoting the input
        input_variables = self.model.get_input()
        # supervised algorithms expect two inputs, feature matrix and label vector
        is_supervised = len(input_variables) > 1
        if is_supervised:
            assert (y is not None)
            if bin_y is not None:
                var_input, var_label, var_label_binary = input_variables
            else:
                var_input, var_label = input_variables
        else:
            # symbolic variable denoting the feature matrix
            var_input = input_variables[0]
            # symbolic variable denoting the label vector (if available)
            var_label = None
            valid_set_y = None
            var_label_binary = None

        # model parameters to be optimized
        params = self.model.parameters()

        # gradient and updates of model's parameters
        gradient, updates = self.model.parameters_gradient_updates(**kwargs)

        # if model does not provide updates, initialize it
        if updates is None:
            updates = OrderedDict()

        if self.momentum > 0:
            # velocity terms for momentum
            velocity = []
            for param in params:
                shape = param.get_value(borrow=True).shape
                vparam = theano.shared(numpy.zeros(shape, dtype=theano.config.floatX), name="{0}_velocity".format(param.name))
                velocity.append(vparam)

            for gparam, param, vparam in zip(gradient, params, velocity):
                # make sure that the learning rate is of the right dtype
                gradient_update = vparam * T.cast(self.momentum, dtype=theano.config.floatX) - \
                                  gparam * T.cast(1 - self.momentum, dtype=theano.config.floatX) * \
                                  T.cast(self.learning_rate / var_input.shape[0], dtype=theano.config.floatX)
                updates[param] = param + gradient_update
                updates[vparam] = gradient_update
        else:
            for gparam, param in zip(gradient, params):
                # make sure that the learning rate is of the right dtype
                gradient_update = - gparam * T.cast(self.learning_rate / var_input.shape[0], dtype=theano.config.floatX)
                updates[param] = param + gradient_update

        # monitoring costs of model
        cost = self.model.get_monitoring_cost(updates)

        givens = {var_input: train_set_x[var_index * self.batch_size:(var_index + 1) * self.batch_size]}

        if is_supervised:
            if bin_y != None:
                givens[var_label_binary] = train_set_y_binary[var_index * self.batch_size:(var_index + 1) * self.batch_size]
            if var_label != None:
                givens[var_label] = train_set_y[var_index * self.batch_size:(var_index + 1) * self.batch_size]

        train_func = theano.function([var_index], cost, updates=updates, givens=givens, name="train_model",
                                     on_unused_input='warn')

        self._do_stop = False
        prev_handler = signal.signal(signal.SIGTERM, self._signal_handler)

        for epoch in range(self.n_iter):
            if self._do_stop:
                print("Aborting")
                break

            mean_cost = []
            for batch_index in range(n_train_batches):
                mean_cost.append(train_func(batch_index))

            if self.stopping_criteria is not None and validation_x is not None:
                err = self.stopping_criteria.update(valid_set_x, valid_set_y)
                print('Training epoch {0}, cost is {1:.4f}, validation error is {2}'.format(epoch,
                      numpy.mean(mean_cost), err))

                if self.stopping_criteria.is_satisfied():
                    print('Early stopping reached at epoch{0}, cost is {1:.4f}, validation error is {2}'.format(epoch,
                      numpy.mean(mean_cost), err))
                    break
            else:
                print('Training epoch {0}, cost is {1:.4f}'.format(epoch, numpy.mean(mean_cost)))

        signal.signal(signal.SIGTERM, prev_handler)