#
# def translator_rnatoatac(encoded_data, trans_ver, reuse=tf.AUTO_REUSE):
#     """
#     translate from scRNA to scATAC
#     """
#     with tf.variable_scope('translator_xy', reuse=tf.AUTO_REUSE):
#         if trans_ver == 'linear':
#             translator_output = tf.layers.Dense(self.embed_dim_y, activation=None, name='translator_xy_1')(
#                 encoded_data);
#         elif trans_ver == '1l':
#             translator_output = tf.layers.Dense(self.embed_dim_y, activation=tf.nn.leaky_relu,
#                                                 name='translator_xy_1')(encoded_data);
#         elif trans_ver == '2l':
#             l1 = tf.layers.Dense(self.embed_dim_y, activation=tf.nn.leaky_relu, name='translator_xy_1')(
#                 encoded_data);
#             l2 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True)
#             translator_output = tf.layers.Dense(self.embed_dim_y, activation=tf.nn.leaky_relu,
#                                                 name='translator_xy_2')(l2);
#         return translator_output;
#
# def translator_atactorna(encoded_data, trans_ver, reuse=tf.AUTO_REUSE):
#     """
#     translate from scATAC to scRNA
#     """
#     with tf.variable_scope('translator_yx', reuse=tf.AUTO_REUSE):
#         if trans_ver == 'linear':
#             translator_output = tf.layers.Dense(self.embed_dim_x, activation=None, name='translator_yx_1')(
#                 encoded_data);
#         elif trans_ver == '1l':
#             translator_output = tf.layers.Dense(self.embed_dim_x, activation=tf.nn.leaky_relu,
#                                                 name='translator_yx_1')(encoded_data);
#         elif trans_ver == '2l':
#             l1 = tf.layers.Dense(self.embed_dim_x, activation=tf.nn.leaky_relu, name='translator_yx_1')(
#                 encoded_data);
#             l2 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True)
#             translator_output = tf.layers.Dense(self.embed_dim_x, activation=tf.nn.leaky_relu,
#                                                 name='translator_yx_2')(l2);
#         return translator_output;