# -*- coding: utf-8 -*-
"""
Visual module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import vgg

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
        model_fn = getattr(vgg, self.config.vgg_type)
        
        logits, _ = model_fn(self.image,
                             is_training=(self.mode == "train"),
                             num_classes=self.config.num_predicates,
                             scope=self.config.vgg_type)
        self.vgg_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.config.vgg_type)
        if self.mode == "inference":
            #self.prediction = tf.nn.softmax(logits, name="softmax")
            self.prediction = logits
        else:
            batch_correct = tf.equal(tf.argmax(logits, 1), self.labels)
            batch_accuracy = tf.reduce_mean(tf.cast(batch_correct, tf.float32))
            batch_top5_correct = tf.nn.in_top_k(logits, self.labels, 5)
            batch_top5_accuracy = tf.reduce_mean(tf.cast(batch_top5_correct, tf.float32))
            tf.summary.scalar("losses/batch_accuracy", batch_accuracy)
            tf.summary.scalar("losses/batch_top5_accuracy", batch_top5_accuracy)
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
        self.setup_global_step()
        self.build_inputs()
        self.build_model()
        self.setup_vgg_initializer()
        
