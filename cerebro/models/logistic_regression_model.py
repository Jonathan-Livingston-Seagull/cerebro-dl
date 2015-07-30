import theano
import theano.tensor as T
import numpy


class LogisticRegressionModel(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, var_input, var_label, n_features, n_classes):
        self.var_input = var_input
        self.var_label = var_label
        self.W = theano.shared(numpy.zeros((n_features, n_classes), dtype=theano.config.floatX),
                                     borrow=True, name="W")
        self.b = theano.shared(numpy.zeros((n_classes,), dtype=theano.config.floatX),
                                  borrow=True, name="b")

        self.p_y_given_x = T.nnet.softmax(T.dot(var_input, self.W) + self.b)
        self.shape = (n_features, n_classes)
        self.params = [self.W, self.b]

    def parameters(self):
        return self.W, self.b

    def predict_function(self):
        return T.argmax(self.p_y_given_x, axis=1)

    def get_monitoring_cost(self, updates=None):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        
        cost = -T.mean(T.log(self.p_y_given_x)[T.arange(self.var_label.shape[0]), self.var_label])
        return cost

    def parameters_gradient_updates(self):
        cost = self.get_monitoring_cost()
        return T.grad(cost, self.params), None

    def get_input(self):
        return self.var_input, self.var_label

    def get_validation_error(self):
        return T.mean(T.neq(self.predict_function(), self.var_label))

