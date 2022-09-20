import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Dist(object):
    def __init__(self, size):
        self.size = size
        self.param_size = size


class NormalDist(Dist):
    def __init__(self, size, std=1.):
        super(NormalDist, self).__init__(size)
        self.param_size = size*2
        self.std = tf.Variable(std)
        self.entropy = 0.5*tf.math.log(np.pi*2*std*std)+0.5

    def sample(self, N):
        return tf.random.normal((N, self.size))*self.std 

    @tf.function
    def logP(self, y, x):
        mu, logvar = tf.split(y, num_or_size_splits=[self.size]*2, axis=1)
        return  self.entropy*np.prod(x.shape)+(-(0.5*tf.reduce_sum(tf.math.log(2.*np.pi)
            +logvar) + 0.5*tf.reduce_sum((x-mu)*tf.exp(-logvar)*(x-mu))))



class NormalDistCov(Dist):
    def __init__(self, size, std=1.):
        super(NormalDistCov, self).__init__(size)
        self.mean_size = size
        self.cov_size = (size*(size+1))//2
        self.param_size = self.mean_size + self.cov_size
        self.std = tf.Variable(std)
        self.entropy = 0.5*tf.math.log(np.pi*2*std*std)+0.5
        self.to_tril = tfp.bijectors.FillScaleTriL()

    def sample(self, N):
        return tf.random.normal((N, self.size))*self.std 

    @tf.function
    def logP(self, y, x):
        mu, L = tf.split(y, num_or_size_splits=[self.mean_size, self.cov_size], axis=1)
        L = self.to_tril.forward(L/10.)
        return (self.entropy*np.prod(x.shape) +
                tf.reduce_sum(tfp.distributions.MultivariateNormalTriL(mu,L).log_prob(x)))




class NormalDistSkew(Dist):
    def __init__(self, size, std=1.):
        super(NormalDistSkew, self).__init__(size)
        self.param_size = size*3
        self.std = tf.Variable(std)
        self.entropy = 0.5*tf.math.log(np.pi*2*std*std)+0.5

    def sample(self, N):
        return tf.random.normal((N, self.size))*self.std 

    def logP(self, y, x):
        mu, logvar, lograte = tf.split(y, num_or_size_splits=[self.size]*3, axis=1)
        return  (self.entropy*np.prod(x.shape) +
                tf.reduce_sum(tfp.distributions.GeneralizedNormal(mu, tf.math.exp(logvar/2.), tf.math.exp(lograte/2.)).log_prob(x)))
                #tf.reduce_sum(tfp.distributions.ExponentiallyModifiedGaussian(mu, tf.math.exp(logvar/2.), tf.math.exp(lograte)).log_prob(x)))



class BinaryDist(Dist):
    def sample(self, N):
        return 2*tf.random.uniform((N,self.size),maxval=2,dtype=tf.dtypes.int32)-1

    def logP(self, y, x):
        return (tf.math.log(2.)*np.prod(x.shape) +
            tf.reduce_sum(tf.math.log_sigmoid(tf.cast(x,y.dtype)*y)))
            #tf.reduce_sum(tf.math.log_sigmoid(tf.cast(2*x-1,y.dtype)*y)))

class BinaryDist2(Dist):
    def __init__(self, size):
        super(BinaryDist2, self).__init__(size)
        self.param_size = size*2

    def sample(self, N):
        return 2*tf.random.uniform((N,self.size),maxval=2,dtype=tf.dtypes.int32)-1

    def logP(self, y, x):
        y1,y2 = tf.split(y, num_or_size_splits=[self.size]*2, axis=1)
        return (tf.math.log(2.)*np.prod(x.shape) +
            tf.reduce_sum(
                tf.math.log(
                    0.5*tf.reduce_prod(tf.sigmoid(tf.cast(x,y.dtype)*y1),axis=1) + 
                    0.5*tf.reduce_prod(tf.sigmoid(tf.cast(x,y.dtype)*y2),axis=1)
                    )  
                )
            )



class UniformDist(Dist):
    def __init__(self, size, std=1.):
        super(UniformDist, self).__init__(size)
        self.param_size = size*2

    def sample(self, N):
        return tf.random.uniform((N,self.size),maxval=1,dtype=tf.dtypes.float32)

    def logP(self, y, x):
        a,b = tf.split(tf.exp(y)+1, num_or_size_splits=[self.size]*2, axis=1)
        xr = 0.99998*x + 0.00001
        return tf.reduce_sum( (a-1)*tf.math.log(xr) + (b-1)*tf.math.log(1-xr) +
                tf.math.lgamma(a+b) - tf.math.lgamma(a) - tf.math.lgamma(b))


class BetaDist(Dist):
    def __init__(self, size, alpha=1.):
        super(BetaDist, self).__init__(size)
        self.d = tfp.distributions.Beta(alpha,alpha)
        self.param_size = size*2

    def sample(self, N):
        return self.d.sample((N,self.size))

    def logP(self, y, x):
        a,b = tf.split(tf.exp(y), num_or_size_splits=[self.size]*2, axis=1)
        xr = 0.99998*x + 0.00001
        return (self.d.entropy()*np.prod(x.shape) + 
                tf.reduce_sum( (a-1)*tf.math.log(xr) + (b-1)*tf.math.log(1-xr) +
                tf.math.lgamma(a+b) - tf.math.lgamma(a) - tf.math.lgamma(b)))


