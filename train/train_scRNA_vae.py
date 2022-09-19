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
                 chr_list, nlayer, dropout_rate, output_model, learning_rate_x,
                 hidden_frac=2, kl_weight=1):
        """
        Network architecture and optimization

        Inputs
        ----------
        input_x: scRNA expression, ncell x input_dim_x, float
        batch_x: scRNA batch factor, ncell x batch_dim_x, int
        chr_list: dictionary using chr as keys and corresponding peak index as vals

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
        self.chr_list = chr_list # NL: Do we need this?

        def encoder_rna(input_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA encoder
            Parameters
            ----------
            hidden_frac: used to divide intermediate dimension to shrink the total paramater size to fit into memory
            input_data: generated from tf.concat([self.input_x, self.batch_x], 1), ncells x (input_dim_x + batch_dim_x)
            """
            with tf.variable_scope('encoder_x', reuse=tf.AUTO_REUSE):
                self.intermediate_dim = int(
                    math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x) / hidden_frac)
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_' + str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                encoder_output_mean = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_mean')(l1)
                encoder_output_var = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_var')(l1)
                encoder_output_var = tf.clip_by_value(encoder_output_var, clip_value_min=-2000000, clip_value_max=15)
                encoder_output_var = tf.math.exp(encoder_output_var) + 0.0001
                eps = tf.random_normal((tf.shape(input_data)[0], self.embed_dim_x), 0, 1, dtype=tf.float32)
                encoder_output_z = encoder_output_mean + tf.math.sqrt(encoder_output_var) * eps
                return encoder_output_mean, encoder_output_var, encoder_output_z;

        def decoder_rna(encoded_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA decoder
            Parameters
            ----------
            hidden_frac: intermadiate layer dim, used hidden_frac to shrink the size to fit into memory
            layer_norm_type: how we normalize layer, don't worry about it now
            encoded_data: generated from concatenation of the encoder output self.encoded_x and batch_x: tf.concat([self.encoded_x, self.batch_x], 1), ncells x (embed_dim_x + batch_dim_x)
            """

            self.intermediate_dim = int(math.sqrt((
                                                              self.input_dim_x + self.batch_dim_x) * self.embed_dim_x) / hidden_frac);  ## TODO: similarly, here we need to add another dimension for each new input batch factor (where previous batch dimension will be 0 and new dimension will be 1). Add an option of only fine tune weights coming out of this added batch dimension when we input new dataset.
            with tf.variable_scope('decoder_x', reuse=tf.AUTO_REUSE):
                # self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x));
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_0')(encoded_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_' + str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                px = tf.layers.Dense(self.intermediate_dim, activation=tf.nn.relu, name='decoder_x_px')(l1);
                px_scale = tf.layers.Dense(self.input_dim_x, activation=tf.nn.softmax, name='decoder_x_px_scale')(px);
                px_dropout = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_dropout')(px)
                px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_r')(px)  # , use_bias=False

                return px_scale, px_dropout, px_r



        self.libsize_x = tf.reduce_sum(self.input_x, 1)

        self.px_z_m, self.px_z_v, self.encoded_x = encoder_rna(tf.concat([self.input_x, self.batch_x], 1), self.nlayer,
                                                               self.hidden_frac);

        z = tf.truncated_normal(tf.shape(self.px_z_m), stddev=1.0) # NL: is this used?

        ## scRNA reconstruction
        self.px_scale, self.px_dropout, self.px_r = decoder_rna(tf.concat([self.encoded_x, self.batch_x], 1),
                                                                self.nlayer, self.hidden_frac);
        if self.dispersion == 'genebatch':
            self.px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='px_r_genebatch_x')(self.batch_x)

        self.px_r = tf.clip_by_value(self.px_r, clip_value_min=-2000000, clip_value_max=15)
        self.px_r = tf.math.exp(self.px_r)
        self.reconstr_x = tf.transpose(tf.transpose(self.px_scale) * self.libsize_x)

        ## scRNA loss
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
        self.train_vars_x = [var for var in tf.trainable_variables() if '_x' in var.name];
        self.loss_x = self.reconstr_loss_x + self.kl_weight_x * self.kld_loss_x
        self.optimizer_x = tf.train.AdamOptimizer(learning_rate=self.learning_rate_x, epsilon=0.01).minimize(
            self.loss_x, var_list=self.train_vars_x);

        self.sess.run(tf.global_variables_initializer());
        self.sess = tf.Session();