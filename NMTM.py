import tensorflow as tf
import numpy as np


class NMTM(object):
    def __init__(self, config, Map_en2cn, Map_cn2en):
        self.config = config
        self.Map_en2cn = Map_en2cn
        self.Map_cn2en = Map_cn2en
        self.build_graph()

    def build_graph(self,):
        self.a = 1 * np.ones((1, int(self.config['topic_num']))).astype(np.float32)
        self.mu_priori = tf.constant((np.log(self.a).T - np.mean(np.log(self.a),1)).T)
        self.sigma_priori = tf.constant((((1.0/self.a)*( 1 - (2.0/self.config['topic_num']) ) ).T + \
            (1.0/(self.config['topic_num'] * self.config['topic_num']) )*np.sum(1.0/self.a,1) ).T)

        self.x_cn = tf.placeholder(tf.float32, [None, self.config["vocab_size_cn"]])
        self.x_en = tf.placeholder(tf.float32, [None, self.config["vocab_size_en"]])
        self.learning_rate = tf.placeholder_with_default(self.config['learning_rate'], shape=())
        self.keep_prob = tf.placeholder_with_default(1.0,  shape=())

        self.phi_cn = tf.get_variable("phi_cn", shape=(self.config['topic_num'], self.config['vocab_size_cn']), initializer=tf.contrib.layers.xavier_initializer())
        self.phi_en = tf.get_variable("phi_en", shape=(self.config['topic_num'], self.config['vocab_size_en']), initializer=tf.contrib.layers.xavier_initializer())

        self.W_cn = tf.get_variable('W_cn', shape=(self.config['vocab_size_cn'], self.config['e1']), initializer=tf.contrib.layers.xavier_initializer())
        self.W_en = tf.get_variable('W_en', shape=(self.config['vocab_size_en'], self.config['e1']), initializer=tf.contrib.layers.xavier_initializer())

        self.B_cn = tf.get_variable('B_cn', shape=(self.config['e1']), initializer=tf.zeros_initializer())
        self.B_en = tf.get_variable('B_en', shape=(self.config['e1']), initializer=tf.zeros_initializer())

        self.beta_cn = self.config['lam'] * tf.matmul(self.phi_en, self.Map_en2cn) + (1 - self.config['lam']) * self.phi_cn
        self.beta_en = self.config['lam'] * tf.matmul(self.phi_cn, self.Map_cn2en) + (1 - self.config['lam']) * self.phi_en

        self.z_cn, self.z_mean_cn, self.z_log_sigma_sq_cn = self.encode(self.x_cn, 'cn')
        self.z_en, self.z_mean_en, self.z_log_sigma_sq_en = self.encode(self.x_en, 'en')

        self.x_recon_cn = self.decode(self.z_cn, self.beta_cn)
        self.x_recon_en = self.decode(self.z_en, self.beta_en)

        self.loss_cn = self.get_loss(self.x_cn, self.x_recon_cn, self.z_mean_cn, self.z_log_sigma_sq_cn)
        self.loss_en = self.get_loss(self.x_en, self.x_recon_en, self.z_mean_en, self.z_log_sigma_sq_en)
        self.loss = self.loss_cn + self.loss_en

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.99).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def encode(self, x, lang):
        act_fun = tf.nn.softplus

        W = getattr(self, 'W_{}'.format(lang))
        B = getattr(self, 'B_{}'.format(lang))

        h = act_fun(tf.matmul(x, W) + B)

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            W2 = tf.get_variable('W2', shape=(self.config['e1'], self.config['e2']), initializer=tf.contrib.layers.xavier_initializer())
            B2 = tf.get_variable('B2', shape=(self.config['e2']), initializer=tf.zeros_initializer())

            W_m = tf.get_variable('W_m', shape=(self.config['e2'], self.config['topic_num']), initializer=tf.contrib.layers.xavier_initializer())
            B_m = tf.get_variable('B_m', shape=(self.config['topic_num']), initializer=tf.zeros_initializer())

            W_s = tf.get_variable('W_s', shape=(self.config['e2'], self.config['topic_num']), initializer=tf.contrib.layers.xavier_initializer())
            B_s = tf.get_variable('B_s', shape=(self.config['topic_num']), initializer=tf.zeros_initializer())

            h = act_fun(tf.matmul(h, W2) + B2)
            h = tf.nn.dropout(h, self.keep_prob)
            mean = tf.contrib.layers.batch_norm(tf.matmul(h, W_m) + B_m)
            log_sigma_sq = tf.contrib.layers.batch_norm(tf.matmul(h, W_s) + B_s)

        eps = tf.random_normal((self.config['batch_size'], self.config['topic_num']), 0, 1, dtype=tf.float32)
        z = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), eps))
        z = tf.nn.softmax(z)
        z = tf.nn.dropout(z, self.keep_prob)
        return z, mean, log_sigma_sq

        self.x_recon_cn = self.decode(self.z_cn, self.beta_cn)
        self.x_recon_cn = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.add(tf.matmul(self.z_cn, self.beta_cn), 0.0)))

    def decode(self, z, beta):
        x_recon = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.matmul(z, beta)))
        return x_recon

    def get_loss(self, x, x_recon, z_mean, z_log_sigma_sq):
        sigma = tf.exp(z_log_sigma_sq)
        latent_loss = 0.5 * (tf.reduce_sum(tf.div(sigma, self.sigma_priori), 1) + 
            tf.reduce_sum(tf.multiply(tf.div((self.mu_priori - z_mean), self.sigma_priori),
                  (self.mu_priori - z_mean)), 1) - self.config['topic_num'] + tf.reduce_sum(tf.log(self.sigma_priori), 1) - tf.reduce_sum(z_log_sigma_sq, 1))

        recon_loss = tf.reduce_sum(-x * tf.log(x_recon), axis=1)
        loss = latent_loss + recon_loss
        return loss
