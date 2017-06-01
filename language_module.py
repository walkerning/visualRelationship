# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class LanguageModule(object):
    def __init__(self, config, mode):
        assert mode in {"train", "inference"}
        self.config = config
        self.mode = mode
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

    def setup_embedding(self):
        self.predicate_embeddings = np.fromfile(self.config.predicate_embedding_file).reshape([self.config.num_predicates, self.config.dim_embedding])
        embedding_init = np.fromfile(self.config.embedding_file).reshape([self.config.num_objects, self.config.dim_embedding])
        with tf.device('/cpu:0'):
            self.embeddings = tf.get_variable(
                name="embedding",
                shape=[self.config.num_objects, self.config.dim_embedding],
                trainable=False,
                initializer=tf.constant_initializer(embedding_init)
            )

    def build_K_loss(self):
        ## build K loss
        predicates = tf.random_uniform((self.config.num_K_samples * 2, 1), minval=0, maxval=self.config.num_predicates, dtype=tf.int32)
        objs = tf.random.uniform((1, self.config.num_K_samples * 4), minval=0, maxval=self.config.num_objects, dtype=tf.int32)
        first = tf.concat([predicates[:self.config.num_K_samples], objs[:self.config.num_K_samples * 2].reshape([self.config.num_K_samples, 2])], axis=1)
        second = tf.concat([predicates[self.config.num_K_samples:], objs[self.config.num_K_samples * 2:].reshape([self.config.num_K_samples, 2])], axis=1)
        
        mask = (tf.reduce_sum(tf.cast(tf.not_equal(first, second), tf.int8), axis=1) > 0)
        first = tf.boolean_mask(first, mask)
        second = tf.boolean_mask(second, mask)
        tf.summary.scalar("K/actual_samples", tf.shape(first)[0])
        # calcualte the cosine distance between each pair of <predicate obj1 obj2>
        dists = tf.squeeze(tf.reduce_sum([1 - tf.reduce_sum(tf.nn.l2_normalize(self.predicate_embeddings[first[:, d], :], dim=1) * tf.nn.l2_normalize(self.predicate_embeddings[second[:, d], :], dim=1)) for d in range(3)], axis=0))
        first_fscores = tf.reduce_sum(self.f_scores(first[:, 1], first[:, 2]) * tf.one_hot(first[:, 0], self.config.num_predicates), axis=1)
        second_fscores = tf.reduce_sum(self.f_scores(second[:, 1], second[:, 2]) * tf.one_hot(second[:, 0], self.config.num_predicates), axis=1)
        _, K_loss = tf.nn.moments(tf.square(tf.squeeze(first_fscores - second_fscores)) / dists, 0)
        tf.losses.add_loss(K_loss)
        tf.summary.scalar("losses/K_loss", self.config.coeff_K * K_loss)

    def build_L_loss(self):
        # L loss需要处理一遍数据集. 看看没个对出现的次数. 然后做negative sample也是实现构建好samples的loss
        with open(self.config.positive_relations_file, "r") as f:
            parsed = [line.split("\t") for line in f.readlines()]
            parsed = [((int(ind) for ind in line[0].split(" ")), int(line[1])) for line in parsed]
        relation_keys = [r[0] for r in parsed]
        # A simple hash key list for each relation triple (pred, obj1, obj2)
        keys = [r[0][0] * 10000 + r[0][1] * 100 + r[0][2] for r in parsed]
        values = [r[1] for r in parsed]
        # The occurence hash table of positive relations
        occur_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys, values), 
            default_value=0,
            name="relation_occur_table"
        )
        # Sample postive samples
        self.L_sample_logits = np.array([values], type=np.float) / self.config.L_sample_temperature
        postive_samples = tf.multinomial(self.L_sample_logits, self.config.num_L_samples)
        postive_occurs = tf.gather(values, positive_samples)
        postive_samples = tf.gather(relation_keys, postive_samples)
        # Randomly construct negative samples
        negative_obj1 = tf.random_uniform((self.config.num_L_samples, 1), minval=0, maxval=self.config.num_objects, dtype=tf.int32)
        negative_obj2 = tf.random_uniform((self.config.num_L_samples, 1), minval=0, maxval=self.config.num_objects, dtype=tf.int32)
        negative_preds = tf.random_uniform((self.config.num_L_samples, 1), minval=0, maxval=self.config.num_predicates, dtype=tf.int32)
        negative_samples = tf.concat([negative_preds, negative_obj1, negative_obj2], axis=1)
        negative_keys = negative_preds * 10000 + negative_obj1 * 100 + negative_obj2

        # Calculate the mask, filtering some wrong negative samples
        negative_occurs = occur_table.lookup(negative_keys)
        mask = tf.logic_or(negative_occurs == 0, positive_occurs - negative_occurs > self.config.min_pos_neg_diff)

        positive_samples = tf.boolean_mask(postive_samples, mask)
        negative_samples = tf.boolean_mask(negative_samples, mask)
        tf.summary.scalar("L/actual_samples", tf.shape(positive_samples)[0])
        # Get fscores for postive/negative samples
        pos_fscores = tf.reduce_sum(self.f_scores(positive_samples[:, 1], positive_samples[:, 2]) * 
                                    tf.one_hot(postive_samples[:, 0], self.config.num_predicates), 
                                    axis=1)
        neg_fscores = tf.reduce_sum(self.f_scores(negative_samples[:, 1], negative_samples[:, 2]) *
                                    tf.one_hot(negative_samples[:, 0], self.config.num_predicates),
                                    axis=1)
        # Ranking loss: award frequently-occuring relations
        L_loss = tf.reduce_sum(tf.maximum(neg_fscores - pos_fscores + 1, 0))
        tf.losses.add_loss(L_loss)
        tf.summary.scalar("losses/L_loss", self.config.coeff_L * L_loss)

    def build_C_loss(self):
        # 每张图片是单独还是across图片... acorss图片就只能采样了...
        # 暂且认为每张图片单独
        # records的格式应该是每张图片给true relations一堆并且带上v_score, false relation一堆并且带上v_score
        # 对于一定量的图片算一个C loss.
        pass

    def f_scores(self, obj1, obj2):
        with tf.device('/cpu:0'):
            obj1_embedding = tf.nn.embedding_lookup(self.embeddings, obj1)
            obj2_embedding = tf.nn.embedding_lookup(self.embeddings, obj2)
        whole_embedding = tf.concat([obj1_embedding, obj2_embedding], 1)
        # a fully connected layer
        f_scores = tf.contrib.layers.fully_connected(
            inputs=whole_embedding,
            num_outputs=self.config.num_predicates,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None)
        return f_scores
            

    def build(self):
        self.setup_embedding()
        self.build_K_loss()
        self.build_L_loss()
        self.build_C_loss()
        self.build_model()
