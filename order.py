from IVAE import *
from InputDistributions import *

class orderVAE(IVAE):

    def __init__(self, encoder, decoder, latent_distribution, e1, e2, beta, train=False):
        super(orderVAE, self).__init__(encoder, decoder, latent_distribution)
        self.beta = tf.Variable(beta, trainable = train)
        self.e1 = e1
        self.e2 = e2

    def encode(self, z):
        return self.encoder(z)/8.

    @tf.function
    def Energy(self, x):
        #ans = tf.reduce_sum(tf.math.log_sigmoid((tf.gather(x,self.e1,axis=1)-tf.gather(x,self.e2,axis=1))/0.01))
        ans = tf.reduce_sum(tf.math.sigmoid((tf.gather(x,self.e1,axis=1)-tf.gather(x,self.e2,axis=1))/0.01))
        return ans

    @tf.function
    def Energy2(self, phi):
        loga,logb = tf.split(phi,2,axis=1)
        a,b = tf.exp(loga), tf.exp(logb)

        a1 = tf.gather(a,self.e1,axis=1)
        b1 = tf.gather(b,self.e1,axis=1)

        a2 = tf.gather(a,self.e2,axis=1)
        b2 = tf.gather(b,self.e2,axis=1)

        p1 = tfp.distributions.Beta(a1,b1)
        p2 = tfp.distributions.Beta(a2,b2)

        dx = 0.051578947368421
        ans = 0.
        for xx in tf.linspace(0.01,0.99,20):
            ans += tf.reduce_sum((p2.prob(xx)*(1-p1.cdf(xx))) * dx)
        return ans


    @tf.function
    def Entropy(self, phi):
        loga,logb = tf.split(phi,2,axis=1)
        d = tfp.distributions.Beta(tf.exp(loga),tf.exp(logb))
        return tf.reduce_sum(d.entropy())

    def reparameterize(self, phi, Gumbel_temp=0.):
        loga,logb = tf.split(phi,2,axis=1)
        a,b = tf.exp(loga), tf.exp(logb)
        u1 = tf.random.uniform(a.shape)*0.99998 + 0.00001
        u2 = tf.random.uniform(a.shape)*0.99998 + 0.00001
        X1 = tfp.distributions.Gamma(a,log_rate=0).quantile(u1)
        X2 = tfp.distributions.Gamma(b,log_rate=0).quantile(u2)
        return X1/(X1+X2)

    @tf.function
    def compute_terms(self, z, Gumbel_temp=0.):
        phi = self.encode(z)
        x = self.reparameterize(phi)
        theta = self.decode(x)
        U = self.Energy(x)
        #U = self.Energy2(phi)
        S = self.Entropy(phi)
        ERR = -self.logP(theta, z)
        return U, S, ERR, ERR-S+self.beta*U


if __name__=="__main__":

    import pickle
    S1 = pickle.load(open('samps.p','rb'))
    mu1 = np.array([np.mean(np.argwhere(S1==i)[:,1]) for i in range(32)])

    import networkx as nx
    G = nx.read_edgelist('comparisons.txt',nodetype=np.int32,create_using=nx.DiGraph)
    e1,e2 = [],[]
    for i,j in G.edges():
        e1.append(i)
        e2.append(j)
    e1 = tf.convert_to_tensor(e1)
    e2 = tf.convert_to_tensor(e2)
    #num_nodes = G.number_of_nodes()
    num_nodes = 32

    input_dim = 16

    initializer = tf.keras.initializers.LecunNormal
    activation_fn = 'selu'
    D = BinaryDist(input_dim)

    decoder = tf.keras.Sequential( [
            tf.keras.layers.InputLayer(input_shape=(num_nodes)),
            tf.keras.layers.Flatten(),
            #tf.keras.layers.Dense(hidden_width*num_nodes,activation=activation_fn,kernel_initializer=initializer),
            tf.keras.layers.Dense(1024,activation=activation_fn,kernel_initializer=initializer),
            tf.keras.layers.Dense(D.param_size,kernel_initializer=initializer),
            ] )
    
    encoder = tf.keras.Sequential( [
            tf.keras.layers.InputLayer(input_shape=(input_dim)),
            #tf.keras.layers.Dense(units=(hidden_width*num_nodes),activation=activation_fn,kernel_initializer=initializer),
            tf.keras.layers.Dense(1024,activation=activation_fn,kernel_initializer=initializer),
            tf.keras.layers.Dense(units=(num_nodes*2),kernel_initializer=initializer),
            ] )


    model = orderVAE(encoder, decoder, D, e1, e2, 1.0986122886681098, train=False)
    optimizer = tf.keras.optimizers.Adam(1e-3, clipnorm=10.)

    H = []
    
    for i in range(500): 
        model.train_step(4000, optimizer, 0.)
        H.append(model.test(2000, 0.))
        print(H[-1])
        S2 = np.argsort(model.sample(1000),axis=1)
        mu2 = np.array([np.mean(np.argwhere(S2==i)[:,1]) for i in range(32)])
        print(np.mean((mu1-mu2)**2))


