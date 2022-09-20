from IVAE import *
from InputDistributions import *

class SBMVAE(IVAE):

    def __init__(self, encoder, decoder, latent_distribution, e1, e2, w_in, w_out, trainable=False):
        super(SBMVAE, self).__init__(encoder, decoder, latent_distribution)
        self.e1 = e1
        self.e2 = e2
        self.win_lr  = tf.Variable(tf.math.log(w_in/(1-w_in)), trainable=trainable)
        self.wout_lr = tf.Variable(tf.math.log(w_out/(1-w_out)), trainable=trainable)

    @tf.function
    def Energy(self, phi):
    
        mean_spin = tf.sigmoid(phi) - tf.sigmoid(-phi)
        m_in = tf.reduce_sum((tf.gather(mean_spin,self.e1,axis=1)*tf.gather(mean_spin,self.e2,axis=1)+1)/2,axis=1)
        m_out = -m_in + len(self.e1)
    
        EnA = tf.reduce_sum(tf.sigmoid(phi),axis=1)
        EnB = tf.reduce_sum(tf.sigmoid(-phi),axis=1)
        n = EnA+EnB
        EnA2 = 0.5*(EnA*EnA - tf.reduce_sum(tf.sigmoid(phi)**2,axis=1))
        EnB2 = 0.5*(EnB*EnB - tf.reduce_sum(tf.sigmoid(-phi)**2,axis=1))
    
        ans = tf.reduce_sum(m_in*self.win_lr) + tf.reduce_sum(m_out*self.wout_lr)
        ans += tf.reduce_sum((EnA2+EnB2)*tf.math.log_sigmoid(-self.win_lr))
        ans += tf.reduce_sum((n*(n-1)*0.5 - EnA2 - EnB2)*tf.math.log_sigmoid(-self.wout_lr))
        return -tf.reduce_sum(ans)

#    @tf.function
#    def Energy(self, phi):
#        win = tf.sigmoid(self.win_lr)
#        wout = tf.sigmoid(self.wout_lr)
#        J0 = tf.math.log( wout / (1-wout) )
#        J1 = tf.math.log( win / (1-win) )
#        fin = tf.math.log(1-win)
#        fout = tf.math.log(1-wout)
#    
#        mean_spin = tf.sigmoid(phi) - tf.sigmoid(-phi)
#        m_in = tf.reduce_sum((tf.gather(mean_spin,self.e1,axis=1)*tf.gather(mean_spin,self.e2,axis=1)+1)/2,axis=1)
#        m_out = -m_in + len(self.e1)
#    
#        EnA = tf.reduce_sum(tf.sigmoid(phi),axis=1)
#        EnB = tf.reduce_sum(tf.sigmoid(-phi),axis=1)
#        n = EnA+EnB
#        EnA2 = 0.5*(EnA*EnA - tf.reduce_sum(tf.sigmoid(phi)**2,axis=1))
#        EnB2 = 0.5*(EnB*EnB - tf.reduce_sum(tf.sigmoid(-phi)**2,axis=1))
#    
#        ans = tf.reduce_sum(m_in*J1) + tf.reduce_sum(m_out*J0)
#        ans += tf.reduce_sum((EnA2+EnB2)*fin)
#        ans += tf.reduce_sum((n*(n-1)*0.5 - EnA2 - EnB2)*fout)
#        return -tf.reduce_sum(ans)


if __name__=="__main__":

    import networkx as nx
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G)
    e1,e2 = [],[]
    for i,j in G.edges():
        e1.append(i)
        e2.append(j)
    e1 = tf.convert_to_tensor(e1)
    e2 = tf.convert_to_tensor(e2)

    num_nodes = G.number_of_nodes()
    input_dim = 2
    hidden_width = 32

    D = NormalDist(input_dim, std=1.)

    initializer = tf.keras.initializers.LecunNormal
    activation_fn = 'selu'

    decoder = tf.keras.Sequential( [
            tf.keras.layers.InputLayer(input_shape=(num_nodes)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hidden_width*num_nodes,activation=activation_fn,kernel_initializer=initializer),
            tf.keras.layers.Dense(D.param_size,kernel_initializer=initializer),
            ] )
    
    encoder = tf.keras.Sequential( [
            tf.keras.layers.InputLayer(input_shape=(input_dim)),
            tf.keras.layers.Dense(units=(hidden_width*num_nodes),activation=activation_fn, kernel_initializer=initializer),
            tf.keras.layers.Dense(units=(num_nodes),kernel_initializer=initializer),
            ] )
    

    model = SBMVAE(encoder, decoder, D, e1, e2, 0.3, 0.1, trainable=True)
    optimizer = tf.keras.optimizers.Adam(1e-3, clipnorm=1.)

    H = []
    GT = tf.Variable(0.1)

    for i in range(1000): 
        model.train_step(4000, optimizer, GT)
        H.append(model.test(2000, GT))
        print(H[-1])

    pos = nx.spring_layout(G)
    def drawkc():
        c = np.round(np.array(model.sample(1))).astype(int)[0]
        nx.draw(G,pos=pos,node_color=c,node_size=50)


