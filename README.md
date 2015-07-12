# cerebro-dl
A deep learning library based on Theano

The minimum set of dependencies are

- Python 2.7 or later
- A suitable C/C++ compiler (GCC is recommended)
- [numpy](http://numpy.org)
- [pandas](http://pandas.pydata.org) 0.15.2 or later
- [scipy](http://scipy.org)
- [scikit-learn](http://scikit-learn.org) 0.16
- [theano](http://deeplearning.net/software/theano/)

It is highly recommended to use Python distribution such as [Anaconda](http://continuum.io/downloads) for easy environment setup.

In addition, optimized version of some packages are available for academic use
at https://store.continuum.io/cshop/academicanaconda

Implementations
===============
Restricted boltzman machines for classification problem

Details of this program are described in the following paper:
        Larochelle, Hugo, and Yoshua Bengio. "Classification using discriminative restricted Boltzmann machines.
        In Proceedings of the 25th international conference on Machine learning, pp. 536-543. ACM, 2008.

Lenet ( One type of convolutional neural network )

        Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
        Gradient-Based Learning Applied to Document
        Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
        http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
