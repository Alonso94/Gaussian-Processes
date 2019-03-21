import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow
from gpflow.test_util import notebook_niter,is_continuous_integration
from scipy.cluster.vq import kmeans2

float_type=gpflow.settings.float_type

iter=notebook_niter(1000)


# put non-gpflow model inside the kernel
# 1. load data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("./data/MNIST_data/",one_hot=False)

class MNIST:
    input_dim=784
    n_classes=10
    x=mnist.train.images.astype(float)
    y=mnist.train.labels.astype(float)[:,None]
    x_test=mnist.test.images.astype(float)
    y_test=mnist.test.labels.astype(float)[:,None]

if is_continuous_integration():
    mask=(MNIST.y <=1).squeeze()
    MNIST.x=MNIST.x[mask][:105,300:305]
    MNIST.y=MNIST.y[mask][:105]
    mask=(MNIST.y_test <=1).squeeze()
    MNIST.x_test=MNIST.x_test[mask][:10,300:305]
    MNIST.y_test=MNIST.y_test[mask][:10]
    MNIST.input_dim=5
    MNIST.n_classes=2

# 2. CNN
def cnn(x,output_dim):
    conv1=tf.layers.conv2d(inputs=tf.reshape(x,[-1,28,28,1]),
                            filter=32,
                            kernel_size=[5,5],
                            padding="same",
                           activation=tf.nn.relu)
    pool1=tf.layers.average_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)
    conv2=tf.layers.conv2d(inputs=pool1,
                           filters=64,
                           kernel_size=[5,5],
                           padding="same",
                           activation=tf.nn.relu)
    pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)
    flat=tf.reshape(pool2,[-1,7*7*64])
    return tf.layers.dense(inputs=flat,units=output_dim,activation=tf.nn.relu)

if is_continuous_integration():
    def cnn(x,output_dim):
        return tf.layers.dense(inputs=tf.reshape(x,[-1,MNIST.input_dim],units=output_dim))

# 3. class
class kernel_with_nn(gpflow.kernels.Kernel):
    def __init__(self,kern,f):
        super().__init__(kern.input_dim)
        self.kern=kern
        self.f=f

    def f(self,x):
        if x is not None:
            with tf.variable_scope('forward',reuse=tf.AUTO_REUSE):
                return self.f(x)

    def get_f_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='forward')

    @gpflow.autoflow([gpflow.settings.float_type, [None, None]])
    def compute_f(self,x):
        return self.f(x)

    def K(self,x,x2=None):
        return self.kern.K(x,x2)

    def Kdiag(self,x):
        return self.kern,Kdiag(x)

class