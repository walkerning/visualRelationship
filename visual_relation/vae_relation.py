#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import cPickle
import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import zhusuan as zs

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from examples import conf
from examples.utils import dataset, save_image_collections

mean = None
std = None
z = None
z_logprob_pi = None
colors = ['b', 'g', 'y', 'c', 'm']
log_ph_z = None
log_px_h = None
def main():
    # # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.random.binomial(1, x_train, size=x_train.shape)
    n_x = x_train.shape[1] # 784

    # Define model parameters
    n_h = 40
    n_z = 10
    n_z_particles = 200
    n_h_particles = 1

    @zs.reuse('model')
    def vae(observed, n, n_x, n_h, n_z, gen=False):
        global mean, std, z_logprob_pi, z
        with zs.BayesianNet(observed=observed) as model:
            z_logprob_pi = tf.get_variable(name="pi",
                                           shape=[n_z],
                                           initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            mean = tf.get_variable(name="mean",
                                   shape=[n_z, n_h],
                                   initializer=tf.random_uniform_initializer(minval=0,
                                                                             maxval=500)
                               )
            std = tf.get_variable(name="std",
                                  shape=[n_z, n_h],
                                  initializer=tf.random_uniform_initializer(minval=10,
                                                                            maxval=10)
                              )
            z = zs.OnehotDiscrete("z", tf.tile(tf.expand_dims(z_logprob_pi, 0), [n, 1]), n_samples=n_z_particles * n_h_particles if not gen else 1)
            mean_of_samples = tf.reshape(tf.matmul(tf.reshape(tf.cast(z, tf.float32), [-1, n_z]), mean), [n_z_particles, n_h_particles, -1, n_h] if not gen else [-1, n_h])
            std_of_samples = tf.reshape(tf.matmul(tf.reshape(tf.cast(z, tf.float32), [-1, n_z]), std), [n_z_particles, n_h_particles, -1, n_h] if not gen else [-1, n_h])
            logstd_of_samples = tf.log(std_of_samples)
            h = zs.Normal("h", mean_of_samples, logstd_of_samples, group_event_ndims=1)
            lx_h = layers.fully_connected(h, 500)
            lx_h = layers.fully_connected(lx_h, 500)
            x_logits = layers.fully_connected(lx_h, n_x, activation_fn=None)
            x = zs.Bernoulli("x", x_logits, group_event_ndims=1)
        return model, x_logits

    weight_initializer=tf.random_normal_initializer(mean=0, stddev=0)
    bias_initializer = tf.constant_initializer(0)
    @zs.reuse('variational')
    def q_net(x, n_h):
        # global mean, std, z_logprob_pi
        with zs.BayesianNet() as variational:
            lh_x = layers.fully_connected(tf.to_float(x), 500)
            lh_x = layers.fully_connected(lh_x, 500)
            h_mean = layers.fully_connected(lh_x, n_h, activation_fn=None)
            h_logstd = layers.fully_connected(lh_x, n_h, activation_fn=None)
            h = zs.Normal("h", h_mean, h_logstd, group_event_ndims=1, n_samples=n_h_particles)
        return variational

    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    n = tf.shape(x)[0]

    def log_joint(observed):
        global log_ph_z, log_px_h
        model, _ = vae(observed, n, n_x, n_h, n_z)
        log_ph_z, log_px_h = model.local_log_prob(['h', 'x'])
        return log_ph_z + log_px_h

        #_, mean, std, z_logprob = vae({}, n, n_x, n_z)
    global_step = tf.Variable(0, trainable=False, name="global_step")
    variational = q_net(x, n_h)
    qh_samples, log_qh = variational.query('h', outputs=True,
                                           local_log_prob=True)

    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint,
                observed={"x": x},
                latent={"h": [qh_samples, log_qh]},
                axis=0))

    optimizer = tf.train.AdamOptimizer(0.00001)
    
    #beta = 0.001
    grads_and_vars = optimizer.compute_gradients(-lower_bound)# + beta * tf.nn.l2_loss(std))
    grads = [g for g, _ in grads_and_vars if g is not None]
    var_ind_dct = {name: ind for ind,name in enumerate([v.name for g, v in grads_and_vars if g is not None])}
    infer = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    ## Generate images
    n_gen = 100 # 10 x 10 grid show
    _, x_logits = vae({}, n_gen, n_x, n_h, n_z, gen=True)
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])

    # Define training parameters
    epoches = 500
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    save_freq = 100

    zvalues_dct = {}
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                #print(sess.run([lower_bound], feed_dict={x: x_batch}))
                ans = sess.run([infer, lower_bound, log_ph_z, log_px_h] + grads,
                                 feed_dict={x: x_batch})
                lb = ans[1]
                lbs.append(lb)

            print('Epoch {}: Lower bound = {}'.format(
                epoch, np.mean(np.array(lbs).astype(np.float64))))

            if epoch % save_freq == 0:
                images, z_values = sess.run([x_gen, z.tensor])
                z_values = np.where(np.squeeze(z_values))[1]
                zvalues_dct[epoch] = (np.squeeze(images), z_values)
                name = "results/vae_homework2_2/vae.epoch.{}.png".format(epoch)
                save_image_collections(images, name)
        saver.save(sess, "results/vae_homework2/finalckpt")

    cPickle.dump(zvalues_dct, open("vae_homework2.pkl", "w"))

if __name__ == "__main__":
    main()
