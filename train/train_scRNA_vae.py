#!/usr/bin/env python

import os, sys, argparse, random
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeCV
import scipy
from scipy.io import mmread
import math

"""
NL: much of this code is due to RAN ZHANG. Adapted from PolarBears= codebase (https://github.com/Noble-Lab/Polarbear.git)

train_scRNA_vae.py builds an autoencoder for each scRNA data

"""


class scRNA_VAE:
    def __init__(self, input_dim_x, batch_dim_x, embed_dim_x, dispersion,
                 nlayer, dropout_rate, learning_rate_x,
                 hidden_frac=2, kl_weight=1):
        """
        Network architecture and optimization

        Inputs
        ----------
        input_x: scRNA expression, ncell x input_dim_x, float
        batch_x: scRNA batch factor, ncell x batch_dim_x, int

        Parameters
        ----------
        kl_weight_x: non-negative value, float
        input_dim_x: #genes, int
        batch_dim_x: dimension of batch matrix in s domain, int
        embed_dim_x: embedding dimension in s VAE, int
        learning_rate_x: scRNA VAE learning rate, float
        nlayer: number of hidden layers in encoder/decoder, int, >=1
        dropout_rate: dropout rate in VAE, float
        dispersion: estimate dispersion per gene&batch: "genebatch" or per gene&cell: "genecell"
        hidden_frac: used to divide intermediate layer dimension, int
        kl_weight: weight of KL divergence loss in VAE, float

        """
        # Assign object features
        self.input_dim_x = input_dim_x
        self.batch_dim_x = batch_dim_x
        self.embed_dim_x = embed_dim_x
        self.learning_rate_x = learning_rate_x
        self.nlayer = nlayer
        self.dropout_rate = dropout_rate
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_dim_x]);
        self.batch_x = tf.placeholder(tf.float32, shape=[None, self.batch_dim_x]);
        self.kl_weight_x = tf.placeholder(tf.float32, None);
        self.kl_weight_y = tf.placeholder(tf.float32, None);
        self.dispersion = dispersion
        self.hidden_frac = hidden_frac
        self.kl_weight = kl_weight
        #self.chr_list = chr_list # NL: Do we need this?

        def encoder_rna(input_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA encoder
            Parameters
            ----------
            hidden_frac: used to divide intermediate dimension to shrink the total paramater size to fit into memory
            input_data: generated from tf.concat([self.input_x, self.batch_x], 1), ncells x (input_dim_x + batch_dim_x)
            """
            with tf.variable_scope('encoder_x', reuse=reuse):
                self.intermediate_dim = int(
                    math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x) / hidden_frac)
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_0')(input_data)
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True)
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate)

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_' + str(layer_i))(l1)
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True)
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate)

                encoder_output_mean = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_mean')(l1)
                encoder_output_var = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_var')(l1)
                encoder_output_var = tf.clip_by_value(encoder_output_var, clip_value_min=-2000000, clip_value_max=15)
                encoder_output_var = tf.math.exp(encoder_output_var) + 0.0001
                eps = tf.random_normal((tf.shape(input_data)[0], self.embed_dim_x), 0, 1, dtype=tf.float32)
                encoder_output_z = encoder_output_mean + tf.math.sqrt(encoder_output_var) * eps
                return encoder_output_mean, encoder_output_var, encoder_output_z

        def decoder_rna(encoded_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA decoder
            Parameters
            ----------
            hidden_frac: intermediate layer dim, used hidden_frac to shrink the size to fit into memory
            layer_norm_type: how we normalize layer, don't worry about it now
            encoded_data: generated from concatenation of the encoder output self.encoded_x and batch_x: tf.concat([self.encoded_x, self.batch_x], 1), ncells x (embed_dim_x + batch_dim_x)
            """

            self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x) / hidden_frac)  ## TODO: similarly, here we need to add another dimension for each new input batch factor (where previous batch dimension will be 0 and new dimension will be 1). Add an option of only fine tune weights coming out of this added batch dimension when we input new dataset.

            with tf.variable_scope('decoder_x', reuse=reuse):
                # self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x));
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_0')(encoded_data)
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True)
                l1 = tf.nn.leaky_relu(l1)
                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_' + str(layer_i))(l1)
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True)
                    l1 = tf.nn.leaky_relu(l1)
                px = tf.layers.Dense(self.intermediate_dim, activation=tf.nn.relu, name='decoder_x_px')(l1)
                px_scale = tf.layers.Dense(self.input_dim_x, activation=tf.nn.softmax, name='decoder_x_px_scale')(px)
                px_dropout = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_dropout')(px)
                px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_r')(px)  # , use_bias=False

                return px_scale, px_dropout, px_r



        self.libsize_x = tf.reduce_sum(self.input_x, 1)

        ##########################
        # Call to encoder function
        ##########################
        self.px_z_m, self.px_z_v, self.encoded_x = encoder_rna(tf.concat([self.input_x, self.batch_x], 1), self.nlayer,
                                                               self.hidden_frac)

        #z = tf.truncated_normal(tf.shape(self.px_z_m), stddev=1.0) # NL: is this used?

        ##########################
        ## scRNA reconstruction
        ##########################
        self.px_scale, self.px_dropout, self.px_r = decoder_rna(tf.concat([self.encoded_x, self.batch_x], 1),
                                                                self.nlayer, self.hidden_frac)
        if self.dispersion == 'genebatch':
            self.px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='px_r_genebatch_x')(self.batch_x)

        self.px_r = tf.clip_by_value(self.px_r, clip_value_min=-2000000, clip_value_max=15)
        self.px_r = tf.math.exp(self.px_r)
        self.reconstr_x = tf.transpose(tf.transpose(self.px_scale) * self.libsize_x)

        ###########################
        ## define scRNA loss
        ###########################

        # reconstr loss
        self.softplus_pi = tf.nn.softplus(-self.px_dropout)
        self.log_theta_eps = tf.log(self.px_r + 1e-8)
        self.log_theta_mu_eps = tf.log(self.px_r + self.reconstr_x + 1e-8)
        self.pi_theta_log = -self.px_dropout + tf.multiply(self.px_r, (self.log_theta_eps - self.log_theta_mu_eps))

        self.case_zero = tf.nn.softplus(self.pi_theta_log) - self.softplus_pi
        self.mul_case_zero = tf.multiply(tf.dtypes.cast(self.input_x < 1e-8, tf.float32), self.case_zero)

        self.case_non_zero = (
                -self.softplus_pi
                + self.pi_theta_log
                + tf.multiply(self.input_x, (tf.log(self.reconstr_x + 1e-8) - self.log_theta_mu_eps))
                + tf.lgamma(self.input_x + self.px_r)
                - tf.lgamma(self.px_r)
                - tf.lgamma(self.input_x + 1)
        )
        self.mul_case_non_zero = tf.multiply(tf.dtypes.cast(self.input_x > 1e-8, tf.float32), self.case_non_zero)

        self.res = self.mul_case_zero + self.mul_case_non_zero
        self.reconstr_loss_x = - tf.reduce_mean(tf.reduce_sum(self.res, axis=1))

        # KL loss
        self.kld_loss_x = tf.reduce_mean(0.5 * (
            tf.reduce_sum(-tf.math.log(self.px_z_v) + self.px_z_v + tf.math.square(self.px_z_m) - 1,
                          axis=1))) * self.kl_weight

        ## optimizers
        self.train_vars_x = [var for var in tf.trainable_variables() if '_x' in var.name]
        self.loss_x = self.reconstr_loss_x + self.kl_weight_x * self.kld_loss_x
        self.optimizer_x = tf.train.AdamOptimizer(learning_rate=self.learning_rate_x, epsilon=0.01).minimize(
            self.loss_x, var_list=self.train_vars_x)

        self.sess.run(tf.global_variables_initializer())
        self.sess = tf.Session()

    def train(self, data_x, batch_x, data_x_val, batch_x_val, data_x_co,
              batch_x_co, nepoch_warmup_x, patience, nepoch_klstart_x,
              output_model, batch_size, nlayer, save_model=False):
        """
        train in four steps, in each step, part of neural network is optimized meanwhile other layers are frozen.
        early stopping based on tolerance (patience) and maximum epochs defined in each step
        sep_train_index: 1: train scATAC autoencoder; 2: train scRNA autoencoder; 3: translate scATAC to scRNA; 4: translate scRNA to scATAC
        n_iter_step1: document the niter for scATAC autoencoder, once it reaches nepoch_klstart_y, KL will start to warm up
        n_iter_step2: document the niter for scRNA autoencoder, once it reaches nepoch_klstart_x, KL will start to warm up

        """
        val_reconstr_x_loss_list = []
        val_kl_x_loss_list = []
        last_improvement = 0
        n_iter_step2 = 0  # # keep track of the number of epochs for nepoch_klstart_x


        iter_list = []
   
        loss_val_check_list = []
        my_epochs_max = {}
        my_epochs_max[1] = 500  # min(round(10000000/data_x.shape[0]), max_epoch)
        my_epochs_max[2] = 500  # min(round(10000000/data_y.shape[0]), max_epoch)
        my_epochs_max[3] = 100
        my_epochs_max[4] = 100
        saver = tf.train.Saver()
        sep_train_index = 2
        for iter in range(1, 2000):
            print('iter ' + str(iter))
            sys.stdout.flush()
            
            
            if n_iter_step2 < nepoch_klstart_x:
                kl_weight_x_update = 0
            else:
                kl_weight_x_update = min(1.0, (n_iter_step2 - nepoch_klstart_x) / float(nepoch_warmup_x))

            iter_list.append(iter)
            iter_list = iter_list
            for batch_id in range(0, data_x.shape[0] // batch_size + 1):
                data_x_i = data_x[
                           (batch_size * batch_id): min(batch_size * (batch_id + 1), data_x.shape[0]), ].todense()
                batch_x_i = batch_x[
                            (batch_size * batch_id): min(batch_size * (batch_id + 1), data_x.shape[0]), ].todense()
                self.sess.run(self.optimizer_x, feed_dict={self.input_x: data_x_i, self.batch_x: batch_x_i,
                                                           self.kl_weight_x: kl_weight_x_update,
                                                           self.kl_weight_y: 0.0})

            n_iter_step2 += 1
            loss_reconstruct_x_val = []
            # loss_kl_x_val = []
            for batch_id in range(0, data_x_val.shape[0] // batch_size + 1):
                data_x_vali = data_x_val[(batch_size * batch_id): min(batch_size * (batch_id + 1),
                                                                      data_x_val.shape[0]), ].todense()
                batch_x_vali = batch_x_val[(batch_size * batch_id): min(batch_size * (batch_id + 1),
                                                                        data_x_val.shape[0]), ].todense()
                loss_reconstruct_x_val_i, loss_kl_x_val_i = self.get_losses_rna(data_x_vali, batch_x_vali, kl_weight_x_update)
                loss_reconstruct_x_val.append(loss_reconstruct_x_val_i)
                # loss_kl_x_val.append(loss_kl_x_val_i)

            loss_val_check = np.nanmean(np.array(loss_reconstruct_x_val))
            val_reconstr_x_loss_list.append(loss_val_check)
            # val_kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x_val)))

            if np.isnan(loss_reconstruct_x_val).any():
                break

           
            if ((iter + 1) % 1 == 0):  # check every epoch
                print('loss_val_check: ' + str(loss_val_check))
                loss_val_check_list.append(loss_val_check)
                try:
                    loss_val_check_best
                except NameError:
                    loss_val_check_best = loss_val_check
                if loss_val_check < loss_val_check_best:
                    # save_sess = self.sess
                    saver.save(self.sess, output_model + '_step' + str(sep_train_index) + '/mymodel')
                    loss_val_check_best = loss_val_check
                    last_improvement = 0
                else:
                    last_improvement += 1

                if len(loss_val_check_list) > 1:
                    ## decide on early stopping
                    stop_decision = last_improvement > patience
                    if stop_decision or len(iter_list) == my_epochs_max[sep_train_index] - 1:
                        tf.reset_default_graph()
                        saver = tf.train.import_meta_graph(
                            output_model + '_step' + str(sep_train_index) + '/mymodel.meta')
                        saver.restore(self.sess,
                                      tf.train.latest_checkpoint(output_model + '_step' + str(sep_train_index) + '/'))
                        print('step' + str(sep_train_index) + ' reached minimum, switching to next')
                        last_improvement = 0
                        loss_val_check_list = []
                        del loss_val_check_best
                        sep_train_index += 1
                        if sep_train_index > 4:
                            break

        return iter_list, val_reconstr_x_loss_list, val_kl_x_loss_list

    def load(self, output_model):
        """
        load pre-trained model
        """
        saver = tf.train.Saver()
        if os.path.exists(output_model + '_step4/'):
            print('== load existing model from ' + output_model + '_step4/')
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph(output_model + '_step4/mymodel.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(output_model + '_step4/'))

    def predict_embedding(self, data_x, batch_x):
        """
        return scRNA and scATAC projections on VAE embedding layers
        """
        return self.sess.run([self.encoded_x],
                             feed_dict={self.input_x: data_x, self.batch_x: batch_x})
    
    
    def get_losses_rna(self, data_x, batch_x, kl_weight_x):
        """
        return scRNA reconstruction loss
        """
        return self.sess.run([self.reconstr_loss_x, self.kld_loss_x], 
                             feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.kl_weight_x: kl_weight_x})
