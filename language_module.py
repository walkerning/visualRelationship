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
        predicate_embedding_init = np.fromfile(self.config.predicate_embedding_file, dtype=np.float32).reshape([self.config.num_predicates, self.config.dim_embedding])
        embedding_init = np.fromfile(self.config.embedding_file, dtype=np.float32).reshape([self.config.num_objects, self.config.dim_embedding])
        with tf.device('/cpu:0'):
            self.predicate_embeddings = tf.get_variable(
                name="predicate_embedding",
                shape=[self.config.num_predicates, self.config.dim_embedding],
                trainable=False,
                initializer=tf.constant_initializer(predicate_embedding_init)
            )
            self.embeddings = tf.get_variable(
                name="embedding",
                shape=[self.config.num_objects, self.config.dim_embedding],
                trainable=False,
                initializer=tf.constant_initializer(embedding_init)
            )

    def setup_f_scores(self):
        def _f_scores(whole_embedding, num_outputs, weights_initializer):
            # a fully connected layer
            f_scores = tf.contrib.layers.fully_connected(
                inputs=whole_embedding,
                num_outputs=num_outputs,
                activation_fn=None,
                weights_initializer=weights_initializer,
            biases_initializer=None)
            return f_scores
        self._get_f_scores = tf.make_template("get_f_scores", _f_scores, num_outputs=self.config.num_predicates,
                                              weights_initializer=self.initializer)

    def build_K_loss(self):
        ## build K loss
        predicates = tf.random_uniform((self.config.num_K_samples * 2, 1), minval=0, maxval=self.config.num_predicates, dtype=tf.int32)
        objs = tf.random_uniform((self.config.num_K_samples * 4,), minval=0, maxval=self.config.num_objects, dtype=tf.int32)
        first = tf.concat([predicates[:self.config.num_K_samples], tf.reshape(objs[:self.config.num_K_samples * 2], [self.config.num_K_samples, 2])], axis=1)
        second = tf.concat([predicates[self.config.num_K_samples:], tf.reshape(objs[self.config.num_K_samples * 2:], [self.config.num_K_samples, 2])], axis=1)
        
        mask = (tf.reduce_sum(tf.cast(tf.not_equal(first, second), tf.int8), axis=1) > 0)
        first = tf.boolean_mask(first, mask)
        second = tf.boolean_mask(second, mask)
        tf.summary.scalar("K/actual_samples", tf.shape(first)[0])
        # calcualte the cosine distance between each pair of <predicate obj1 obj2>
        tmp_embed_tensors = [self.predicate_embeddings, self.embeddings, self.embeddings]
        dists = tf.squeeze(tf.reduce_sum([1 - tf.reduce_sum(tf.nn.l2_normalize(tf.nn.embedding_lookup(tmp_embed_tensors[d], first[:, d]), dim=1) * 
                                                            tf.nn.l2_normalize(tf.nn.embedding_lookup(tmp_embed_tensors[d], second[:, d]), dim=1)) for d in range(3)],
                                         axis=0))
        first_fscores = tf.reduce_sum(self.f_scores(first[:, 1], first[:, 2]) * tf.one_hot(first[:, 0], self.config.num_predicates), axis=1)
        second_fscores = tf.reduce_sum(self.f_scores(second[:, 1], second[:, 2]) * tf.one_hot(second[:, 0], self.config.num_predicates), axis=1)
        _, K_loss = tf.nn.moments(tf.square(tf.squeeze(first_fscores - second_fscores)) / dists, [0])
        tf.losses.add_loss(K_loss)
        tf.summary.scalar("losses/K_loss", self.config.coeff_K * K_loss)

    def build_L_loss(self):
        # L loss需要处理一遍数据集. 看看没个对出现的次数. 然后做negative sample也是实现构建好samples的loss
        with open(self.config.positive_relations_file, "r") as f:
            parsed = [line.split("\t") for line in f.readlines()]
            parsed = [(tuple([int(ind) for ind in line[0].split(" ")]), int(line[1])) for line in parsed]
        relation_keys = [r[0] for r in parsed]
        # A simple hash key list for each relation triple (pred, obj1, obj2)
        keys = [r[0][0] * 10000 + r[0][1] * 100 + r[0][2] for r in parsed]
        values = [r[1] for r in parsed]
        # The occurence hash table of positive relations
        with tf.device("/cpu:0"):
            occur_table = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, 
                                                            values, 
                                                            key_dtype=tf.int64,
                                                            value_dtype=tf.int64),
                default_value=0,
                name="relation_occur_table"
            )
        # Sample positive samples
        self.L_sample_logits = np.array([values], dtype=np.float32) / self.config.L_sample_temperature
        positive_samples = tf.multinomial(self.L_sample_logits, self.config.num_L_samples)
        positive_occurs = tf.gather(np.array(values, dtype=np.int64), positive_samples)
        positive_samples = tf.squeeze(tf.gather(relation_keys, positive_samples))
        # Randomly construct negative samples
        negative_obj1 = tf.random_uniform((self.config.num_L_samples, 1), minval=0, maxval=self.config.num_objects, dtype=tf.int64)
        negative_obj2 = tf.random_uniform((self.config.num_L_samples, 1), minval=0, maxval=self.config.num_objects, dtype=tf.int64)
        negative_preds = tf.random_uniform((self.config.num_L_samples, 1), minval=0, maxval=self.config.num_predicates, dtype=tf.int64)
        negative_samples = tf.concat([negative_preds, negative_obj1, negative_obj2], axis=1)
        negative_keys = negative_preds * 10000 + negative_obj1 * 100 + negative_obj2

        # Calculate the mask, filtering some wrong negative samples
        negative_occurs = tf.squeeze(occur_table.lookup(negative_keys))
        mask = tf.squeeze(tf.logical_or(negative_occurs == 0, positive_occurs - negative_occurs > self.config.min_pos_neg_diff))

        positive_samples = tf.boolean_mask(positive_samples, mask)
        negative_samples = tf.boolean_mask(negative_samples, mask)
        tf.summary.scalar("L/actual_samples", tf.shape(positive_samples)[0])
        # Get fscores for positive/negative samples
        pos_fscores = tf.reduce_sum(self.f_scores(positive_samples[:, 1], positive_samples[:, 2]) * 
                                    tf.one_hot(positive_samples[:, 0], self.config.num_predicates), 
                                    axis=1)
        neg_fscores = tf.reduce_sum(self.f_scores(negative_samples[:, 1], negative_samples[:, 2]) *
                                    tf.one_hot(negative_samples[:, 0], self.config.num_predicates),
                                    axis=1)
        # Ranking loss: award frequently-occuring relations
        L_loss = tf.reduce_sum(tf.maximum(neg_fscores - pos_fscores + 1, 0))
        tf.losses.add_loss(L_loss)
        tf.summary.scalar("losses/L_loss", self.config.coeff_L * L_loss)

    def build_C_loss(self):
        if self.config.C_cross_image:
            # FIXME: 这里并不完全与论文一样...论文也讲的不清楚
            # 这里先试试不加triple要不等于的约束
            with open(self.config.visual_scores_file, "r") as f:
                lines = [line for line in f.readlines() if line.strip()]

            lines = [line.split(" ")[1:] for line in lines]
            pos_lines = [line[1:] for line in lines if int(line[0]) == 1]
            neg_lines = [line[1:] for line in lines if int(line[0]) == 0]

            positives = np.array(pos_lines)
            pos_pred = positives[:, 0].astype(np.int32)
            pos_obj1 = positives[:, 1].astype(np.int32)
            pos_obj2 = positives[:, 2].astype(np.int32)
            pos_vscores = positives[:, 3].astype(np.float32)

            negatives = np.array(neg_lines).astype(np.float32)
            neg_pred = negatives[:, 0].astype(np.int32)
            neg_obj1 = negatives[:, 1].astype(np.int32)
            neg_obj2 = negatives[:, 2].astype(np.int32)
            neg_vscores = negatives[:, 3].astype(np.float32)

            # calculate max negative scores
            neg_fscores = tf.reduce_sum(self.f_scores(neg_obj1, neg_obj2) * 
                                        tf.one_hot(neg_pred, self.config.num_predicates),
                                        axis=1)
            max_neg_fvscore = tf.reduce_max(neg_fscores * neg_vscores)
            pos_fscores = tf.reduce_sum(self.f_scores(pos_obj1, pos_obj2) * 
                                        tf.one_hot(pos_pred, self.config.num_predicates),
                                        axis=1)
            # Ranking loss
            C_loss = tf.reduce_sum(tf.maximum(max_neg_fvscore - pos_fscores + 1, 0))
            tf.losses.add_loss(C_loss)
            tf.summary.scalar("losses/C_loss", C_loss)

    def f_scores(self, obj1, obj2):
        with tf.device('/cpu:0'):
            obj1_embedding = tf.nn.embedding_lookup(self.embeddings, obj1)
            obj2_embedding = tf.nn.embedding_lookup(self.embeddings, obj2)
        whole_embedding = tf.concat([obj1_embedding, obj2_embedding], 1)
        return self._get_f_scores(whole_embedding)

    def setup_global_step(self):
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    def build(self):
        self.setup_global_step()
        self.setup_embedding()
        self.setup_f_scores()
        self.build_K_loss()
        self.build_L_loss()
        self.build_C_loss()
        self.total_loss = tf.losses.get_total_loss()
