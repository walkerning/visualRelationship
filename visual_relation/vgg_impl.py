# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.slim.python.slim.nets import vgg

def _custom_pool_vgg_parse(name):
    args = name.strip().split()
    kwargs = {}
    if len(args) > 0:
        kwargs["conv_size"] = int(args[0])
    if len(args) > 1:
        kwargs["l1_reg_scale"] = float(args[1])
    return kwargs

def custom_pool_vgg_factory(conv_size=3, l1_reg_scale=0., vgg_name="vgg_16", endname="pool5"):
    inner_vgg = getattr(vgg, vgg_name)
    def _custom_pool_vgg(inputs,
                         num_classes=1000,
                         is_training=True,
                         dropout_keep_prob=0.5,
                         spatial_squeeze=True,
                         scope=vgg_name):
        _, endpoints = inner_vgg(inputs, num_classes, is_training, dropout_keep_prob, spatial_squeeze, scope)
        endpoint = endpoints[scope + "/" + endname]
        # use VALID padding?
        net = layers.conv2d(endpoint,
                            num_classes,
                            [conv_size, conv_size],
                            activation_fn=None,
                            normalizer_fn=None,
                            scope="conv6",
                            weights_regularizer=layers.l1_regularizer(l1_reg_scale))
        # net = layers.conv2d(endpoint,
        #                     num_classes,
        #                     [conv_size, conv_size],
        #                     scope="conv6",
        #                     weights_regularizer=layers.l1_regularizer(l1_reg_scale))
        # net = layers.conv2d(net,
        #                     num_classes,
        #                     [conv_size, conv_size],
        #                     scope="conv7",
        #                     activation_fn=None,
        #                     normalizer_fn=None,
        #                     weights_regularizer=layers.l1_regularizer(l1_reg_scale))
        return tf.reduce_max(net, axis=[1, 2]), endpoints
    return _custom_pool_vgg

def get_model_fn(type_name):
    if type_name.startswith("vgg"):
        return getattr(vgg, type_name)
    else:
        tpname, argname = type_name.split(":", 1)
        tpregistry = custom_factory_dct[tpname]
        return tpregistry[0](**tpregistry[1](argname))

custom_factory_dct = {
    "custom_pool_vgg": (custom_pool_vgg_factory, _custom_pool_vgg_parse)
}
