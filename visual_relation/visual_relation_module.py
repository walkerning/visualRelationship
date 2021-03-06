# -*- coding: utf-8 -*-
"""
nVisual module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from vgg_impl import get_model_fn

class VisualModule(object):
    def __init__(self, config, mode):
        assert mode in {"train", "inference"}
        self.mode = mode
        self.config = config
        self.reader = tf.TFRecordReader()
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        self.init_fn = None
        self.global_step = None

    def image_summary(self, name, image):
        tf.summary.image(name, tf.expand_dims(image, 0))

    def process_image(self, im_str):
        image = tf.reshape(tf.decode_raw(im_str, out_type=tf.uint8), (self.config.image_height, self.config.image_width, 3))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, 
                                       size=[self.config.image_height, self.config.image_width],
                                       method=tf.image.ResizeMethod.BILINEAR)
        return image


    def setup_embeddings(self):
        if self.config.use_predicate_embedding:
            predicate_embedding_init = np.fromfile(self.config.predicate_embedding_file, dtype=np.float32).reshape([self.config.num_predicates, self.config.dim_embedding])
            if self.config.use_pca_embeddings:
                from sklearn import decomposition
                pca = decomposition.PCA(n_components=self.config.dim_pca_embeddings)
                pca.fit(predicate_embedding_init)
                predicate_embedding_init = pca.transform(predicate_embedding_init)

            self.predicate_embeddings = tf.get_variable(
                name="predicate_embedding",
                shape=[self.config.num_predicates, predicate_embedding_init.shape[1]],
                trainable=False,
                initializer=tf.constant_initializer(predicate_embedding_init),
                dtype=tf.float32
            )

    def build_inputs(self):
        if self.mode == "train":
            data_files = []
            for pattern in self.config.file_pattern.split(","):
                data_files.extend(tf.gfile.Glob(pattern))
            print("number of training records files: {}".format(len(data_files)))
            filename_queue = tf.train.string_input_producer(
                data_files,
                shuffle=True,
                capacity=16
            )
            _, example = self.reader.read(filename_queue)
            proto_value = tf.parse_single_example(example,
                                           features={
                                               self.config.image_feature_name: tf.FixedLenFeature([], dtype=tf.string),
                                               self.config.predicate_feature_name: tf.FixedLenFeature([], dtype=tf.int64)
                                           })
            image = proto_value[self.config.image_feature_name]
            #image = tf.image.decode_jpeg(image, channels=3)
            image = self.process_image(image)
            self.image, self.labels = tf.train.batch_join([(image, proto_value[self.config.predicate_feature_name])], batch_size=self.config.batch_size)
            #tf.summary.histogram("batch_labels", self.labels)
        else:
            self.image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            self.image = tf.expand_dims(self.process_image(self.image_feed), 0)

    def build_model(self):
        model_fn = get_model_fn(self.config.vgg_type)
        
        if not self.config.use_predicate_attention:
            logits, endpoints = model_fn(self.image,
                                         is_training=(self.mode == "train"),
                                         num_classes=self.config.num_predicates,
                                         scope=self.config.vgg_scope)
        else:
            _, endpoints = model_fn(self.image,
                                    is_training=(self.mode == "train"),
                                    num_classes=self.config.num_predicates,
                                    scope=self.config.vgg_scope)
            #tf.contrib.layers.conv2d(endpoints["vgg_16/pool5"], num_output=))
            attend_layer_name = self.config.vgg_scope + "/pool5"
            attend_endpoint = endpoints[attend_layer_name]
            num_locations = (attend_endpoint.shape[1] * attend_endpoint.shape[2]).value
            num_features = attend_endpoint.shape[3].value
            with tf.variable_scope("embedding_to_attention"):
                # `attention` shape is [self.config.num_predicates, L=49]
                if self.config.spatial_attention_activation_fn:
                    sp_attention_act_fn = getattr(tf.nn, self.config.spatial_attention_activation_fn)
                else:
                    sp_attention_act_fn = None

                #attention = tf.nn.relu(tf.matmul(self.predicate_embeddings, embedding_to_attention_weight) + embedding_to_attention_bias)
                if self.config.use_predicate_embedding:
                    # whether to use relu or softmax???
                    attention = tf.contrib.layers.fully_connected(
                        inputs=self.predicate_embeddings,
                        num_outputs=num_locations,
                        activation_fn=sp_attention_act_fn,
                        weights_initializer=self.initializer,
                        biases_initializer=tf.constant_initializer(0),
                        weights_regularizer=tf.contrib.layers.l1_regularizer(self.config.l1_reg_scale))
                else:
                    attention = tf.get_variable(
                        name="attention_weights",
                        shape=[self.config.num_predicates, num_locations],
                        initializer=self.initializer,
                        dtype=tf.float32                        
                    )
                    #attention = tf.nn.relu(attention)
                    if sp_attention_act_fn:
                        attention = sp_attention_act_fn(attention)

                with tf.variable_scope("semantic"):
                    if self.config.use_semantic_attention:
                        semantic_attention = tf.get_variable(
                            name="attention_weights",
                            shape=[self.config.num_predicates, num_features],
                            initializer=self.initializer,
                            dtype=tf.float32
                        )
                        # semantic_attention = tf.nn.sigmoid(semantic_attention)
                        if self.config.semantic_attention_activation_fn:
                            se_attention_act_fn = getattr(tf.nn, self.config.semantic_attention_activation_fn)
                            semantic_attention = se_attention_act_fn(semantic_attention)

                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name):
                    tf.summary.histogram(v.op.name, v)

                for pred in range(self.config.num_predicates):
                    tf.summary.histogram("attention/{}".format(pred), attention[pred, :])
                # normalized attention. ReLU is better in my experiements...
                #n_attention = tf.nn.softmax(attention)
                n_attention = attention
                for pred in range(self.config.num_predicates):
                    tf.summary.histogram("normalized_attention/{}".format(pred), n_attention[pred, :])

            # `attended_features` will be of shape [self.config.batch_size, 1, self.config.num_predicates, 512]
            attended_features = tf.expand_dims(tf.reduce_sum(tf.expand_dims(tf.reshape(attend_endpoint, [self.config.batch_size, num_locations, -1]), 1) * tf.expand_dims(n_attention, -1), axis=2), 1)
            if self.config.use_semantic_attention:
                attended_features = attended_features * semantic_attention

            with tf.variable_scope(self.config.vgg_scope):
                net = tf.contrib.layers.conv2d(
                    attended_features,
                    4096, [1,1],
                    scope="attend_fc6")
                # logits = tf.squeeze(tf.reduce_max(net, axis=-1))
                net = tf.contrib.layers.dropout(
                    net, 0.5, is_training=(self.mode == "train"),
                    scope="dropout6")
                logits = tf.squeeze(tf.contrib.layers.conv2d(
                    net, 1, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope="attend_fc8"))

        self.vgg_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.config.vgg_scope)

        if self.mode == "inference":
            #self.prediction = tf.nn.softmax(logits, name="softmax")
            self.prediction = logits
        else:
            # summary the logits and some activations
            with tf.name_scope("logits"):
                for pred in range(self.config.num_predicates):
                    tf.summary.histogram("logits_{}".format(pred), logits[:, pred])
            for end_name in self.config.summary_endpoints:
                tensor_name = self.config.vgg_scope + "/" + end_name
                if tensor_name in endpoints:
                    tf.summary.histogram("{}/activations".format(end_name), endpoints[tensor_name])
            # summary weights/biases of fc layers:
            fc_variables = [v for v in self.vgg_variables if "fc" in v.op.name]
            for v in fc_variables:
                tf.summary.histogram(v.op.name, v)

            batch_correct = tf.equal(tf.argmax(logits, 1), self.labels)
            batch_accuracy = tf.reduce_mean(tf.cast(batch_correct, tf.float32))
            batch_top5_correct = tf.nn.in_top_k(logits, self.labels, 5)
            batch_top5_accuracy = tf.reduce_mean(tf.cast(batch_top5_correct, tf.float32))
            tf.summary.scalar("losses/batch_accuracy", batch_accuracy)
            tf.summary.scalar("losses/batch_top5_accuracy", batch_top5_accuracy)
            if self.config.use_rank_loss:
                # construct n-dim indexes. [(row_ind, col_ind)...]
                nd_inds = tf.concat((tf.expand_dims(tf.constant(range(self.config.batch_size), dtype=tf.int64), -1),
                                     tf.expand_dims(self.labels, -1)), axis=1)
                label_logits = tf.gather_nd(logits, nd_inds)
                tf.summary.histogram("losses/label_logits", label_logits)
                maxneg_logits = tf.reduce_max(tf.where(tf.equal(logits, tf.expand_dims(label_logits, -1)),
                                                       logits,
                                                       tf.zeros((self.config.batch_size, self.config.num_predicates),
                                                                dtype=tf.float32)),
                                              axis=1)
                losses = tf.maximum(maxneg_logits - label_logits + 1, 0)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits) 
            batch_loss = tf.reduce_sum(losses)
            tf.losses.add_loss(batch_loss)
            tf.summary.scalar("losses/batch_loss", batch_loss)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar("losses/total_loss", self.total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("params/" + var.op.name, var)


    def setup_vgg_initializer(self):
        if self.mode != "inference":
            self.conv_variables = [v for v in self.vgg_variables if "conv" in v.op.name]
            saver = tf.train.Saver(self.conv_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring VGG variables from checkpoint file %s",
                                self.config.vgg_checkpoint_file)
                saver.restore(sess, self.config.vgg_checkpoint_file)

            self.init_fn = restore_fn

    def setup_global_step(self):
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    def build(self):
        if self.mode == "train":
            self.setup_global_step()
        if self.config.use_predicate_attention:
            self.setup_embeddings()
        self.build_inputs()
        self.build_model()
        self.setup_vgg_initializer()
        
