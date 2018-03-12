# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg


class unet(Network):
    def __init__(self):
        Network.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._scope = 'unet'

    def _image_to_head(self, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            conv1 = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                              trainable=False, scope='conv1')
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME', scope='pool1')

            conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3],
                              trainable=False, scope='conv2')
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME', scope='pool2')

            conv3 = slim.repeat(pool2, 2, slim.conv2d, 256, [3, 3],
                              trainable=is_training, scope='conv3')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME', scope='pool3')

            conv4 = slim.repeat(pool3, 2, slim.conv2d, 512, [3, 3],
                              trainable=is_training, scope='conv4')

            upconcat3 = self._up_concat(conv4, conv3, 256, 3)
            upconv3 = slim.repeat(upconcat3, 2, slim.conv2d, 256, [3, 3],
                              trainable=is_training, scope='upconv4')

            upconcat2 = self._up_concat(upconv3, conv2, 128, 2)
            upconv2 = slim.repeat(upconcat2, 2, slim.conv2d, 128, [3, 3],
                              trainable=is_training, scope='upconv4')

            upconcat1 = self._up_concat(upconv2, conv1, 64, 4)
            upconv1 = slim.repeat(upconcat1, 2, slim.conv2d, 256, [3, 3],
                              trainable=is_training, scope='upconv4')

        self._act_summaries.append(net)
        self._layers['head'] = net

        return net

    def _up_concat(self, conv1, conv2, num_filters, name):
        upconv1 = tf.nn.conv2d_transpose(value=conv1,
                                     filter=num_filters,
                                     kernel_size=2,
                                     strides=2,
                                     name='up{}'.format(name))
        concat = tf.concat([upconv1, conv2], axis=-1, name="concat{}".format(name))
        return concat

    def _head_to_tail(self, pool5, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            pool5_flat = slim.flatten(pool5, scope='flatten')
            fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
            if is_training:
                fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                                   scope='dropout6')
            fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
            if is_training:
                fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True,
                                   scope='dropout7')

        return fc7

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == (self._scope + '/fc6/weights:0') or \
                    v.name == (self._scope + '/fc7/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16') as scope:
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv,
                                              self._scope + "/fc7/weights": fc7_conv,
                                              self._scope + "/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                                      self._variables_to_fix[
                                                                                                          self._scope + '/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                                      self._variables_to_fix[
                                                                                                          self._scope + '/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))
