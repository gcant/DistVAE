from IVAE import *
from InputDistributions import *

class SqIsingVAE(IVAE):

    def __init__(self, encoder, decoder, latent_distribution, beta, linear_dimension):
        super(SqIsingVAE, self).__init__(encoder, decoder, latent_distribution)
        self.beta = tf.Variable(beta, trainable = False)
        self.l = linear_dimension

    @tf.function
    def Energy(self, phi):
        mean_spin = tf.sigmoid(phi) - tf.sigmoid(-phi)
        mean_spin = tf.reshape(mean_spin, (mean_spin.shape[0],self.l,self.l))
        ans  = tf.reduce_sum(mean_spin[:,1:,:] * mean_spin[:,:-1,:])
        ans += tf.reduce_sum(mean_spin[:,0,:]*mean_spin[:,-1,:])
        ans += tf.reduce_sum(mean_spin[:,:,1:] * mean_spin[:,:,:-1])
        ans += tf.reduce_sum(mean_spin[:,:,0]*mean_spin[:,:,-1])
        return -ans


if __name__=="__main__":
    ld = 8
    #input_dim = ld*(ld-2)
    input_dim = 12
    hidden_width = 1024
    
    initializer = tf.keras.initializers.LecunNormal
    activation_fn = 'selu'


    #D = NormalDistCov(input_dim, std=1.)
    D = BinaryDist(input_dim)


    decoder = tf.keras.Sequential( [
            tf.keras.layers.InputLayer(input_shape=(ld*ld)),
            tf.keras.layers.Dense(hidden_width,activation=activation_fn,kernel_initializer=initializer),
            tf.keras.layers.Dense(D.param_size,kernel_initializer=initializer),
            ] )
    
    encoder = tf.keras.Sequential( [
            tf.keras.layers.InputLayer(input_shape=(input_dim)),
            tf.keras.layers.Dense(units=hidden_width,activation=activation_fn,kernel_initializer=initializer),
            tf.keras.layers.Dense(units=(ld*ld),kernel_initializer=initializer),
            ] )


    beta = tf.Variable(0.35)
    GT = tf.Variable(0.1)

    model = SqIsingVAE(encoder, decoder, D, beta, ld)


    #learn_s = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 1, 0.99, staircase=False, name=None)
    #optimizer = tf.keras.optimizers.Adam(learn_s, clipnorm=1.)
    #optimizer = tf.keras.optimizers.Adam(1e-3,clipnorm=10.)
    optimizer = tf.keras.optimizers.Adam(1e-3,clipvalue=1.)
    #optimizer = tf.keras.optimizers.SGD(0.001, clipnorm=1.)
    H = []
    Z = all_binary_states(input_dim)*2 - 1
    for i in range(2000): 
        model.train_step(4000, optimizer, GT)
        #model.train_step(len(Z), optimizer, GT, z=Z)
        H.append(model.test(2000, GT))
        print(H[-1])

