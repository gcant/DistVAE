import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class IVAE(tf.keras.Model):

    def __init__(self, encoder, decoder, latent_distribution):
        super(IVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.distribution = latent_distribution
        self.num_latent_vars = self.decoder.input_shape[1]
        self.LogitRNG = tfp.distributions.Logistic(0,1)
        self.beta = tf.Variable(1., trainable=False)

    def encode(self, z):
        return self.encoder(z)

    def decode(self, x):
        return self.decoder(2*x-1)

    @tf.function
    def Energy(self, phi):
        return -tf.reduce_sum(phi**2)/2.

    @tf.function
    def Entropy(self, phi):
        return -tf.reduce_sum( (tf.sigmoid(phi)*tf.math.log_sigmoid(phi) +
              tf.sigmoid(-phi)*tf.math.log_sigmoid(-phi)) )
  
    def reparameterize(self, phi, Gumbel_temp):
        l = self.LogitRNG.sample(phi.shape)
        s = tf.math.log_sigmoid(-phi) - tf.math.log_sigmoid(phi)
        return tf.sigmoid(-(l + s)/Gumbel_temp)

    def sample(self, num_samps, Gumbel_temp=tf.Variable(0.,trainable=False)):
        z = self.distribution.sample(num_samps)
        phi = self.encode(z)
        return self.reparameterize(phi,Gumbel_temp)

    def logP(self, theta, z):
        return self.distribution.logP(theta, z)

    @tf.function
    def compute_terms(self, z, Gumbel_temp):
        phi = self.encode(z)
        U = self.Energy(phi)
        S = self.Entropy(phi)
        theta = self.decode(self.reparameterize(phi, Gumbel_temp))
        ERR = -self.logP(theta, z)
        return U, S, ERR, ERR-S+self.beta*U

    def compute_loss(self, z, Gumbel_temp):
        return self.compute_terms(z, Gumbel_temp)[3]

    def train_step(self, batch_size, optimizer, Gumbel_temp, z=None):
        if z is None:
            z = self.distribution.sample(batch_size)
        with tf.GradientTape() as tape:
          loss = self.compute_loss(z, Gumbel_temp)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def test(self, batch_size, Gumbel_temp):
        z = self.distribution.sample(batch_size)
        return np.array(self.compute_terms(z,Gumbel_temp)) / (batch_size*self.num_latent_vars)



def all_binary_states(n):
    return tf.convert_to_tensor(
            [np.array(list(np.binary_repr(i,width=n)), dtype=np.int32) for i in range(2**n)]
            )
