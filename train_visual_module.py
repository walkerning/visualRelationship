# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from visual_relation_module import VisualModule
from configuration import ModelConfig

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("vgg_checkpoint_file", "",
                       "Path to a pretrained vgg model.")
tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 500000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("save_interval_secs", 60,
                        "Frequency at which checkpoints are saved.")
tf.flags.DEFINE_integer("save_summaries_secs", 10,
                        "The frequency with which summaries are saved, in seconds.")

def main(_):
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.train_dir, "--train_dir is required"
    model_config = ModelConfig()
    model_config.file_pattern = FLAGS.input_file_pattern
    model_config.vgg_checkpoint_file = FLAGS.vgg_checkpoint_file
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)
    model = VisualModule(model_config, mode="train")
    model.build()
    #learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(
        0.001,
        model.global_step,
        decay_steps=3000,
        decay_rate=0.1,
        staircase=True)
    tf.summary.scalar("learning_rate", learning_rate)
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=model_config.optimizer,
        clip_gradients=model_config.clip_gradients)
    saver = tf.train.Saver(max_to_keep=model_config.max_checkpoints_to_keep)

    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver,
        save_interval_secs=FLAGS.save_interval_secs,
        save_summaries_secs=FLAGS.save_summaries_secs)


if __name__ == "__main__":
    tf.app.run()
