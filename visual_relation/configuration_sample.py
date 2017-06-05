# -*- coding: utf-8 -*-
"""
An ugly configuration module, `ModelConfig`, `LanguageModuleConfig` contains data/model/training informations.

@TODO: Maybe refactor in the future.
"""

class ModelConfig(object):
    def __init__(self):
        ## model configs
        self.num_predicates = 70
        self.dim_embedding = 300
        
        self.image_feature_name = "bbox"
        
        self.predicate_feature_name = "predicate"

        self.file_pattern = ""

        # self.vgg_type = "vgg_16"
        self.vgg_type = "custom_pool_vgg:"
        
        self.vgg_scope = "vgg_16"

        # input height width for VGG net
        self.image_height = 224
        self.image_width = 224


        ## training configs
        self.initializer_scale = 0.01

        self.vgg_checkpoint_file = "./data/vgg.ckpt"

        self.clip_gradients = 5.

        self.max_checkpoints_to_keep = 5

        self.optimizer = "Adam"

        self.batch_size = 32

        self.summary_endpoints = ["fc6", "fc7"]

        self.use_rank_loss = False

        self.use_predicate_attention = False # worse than not using attention
        self.use_predicate_embedding = True # better than not using predicate embedding
        self.predicate_embedding_file = "./model/predicate_embeddings.npz"
        
        self.use_semantic_attention = False # indifferent of not using semantic attention

        self.l1_reg_scale = 0.

        # preprocessing
        self.use_pca_embeddings = False # if use_pca_embeddings, more stable, but the performance is of no better
        self.dim_pca_embeddings = 50

        # self.attention_activation_fn = "relu" # sigmoid / relu / softmax
        self.spatial_attention_activation_fn = "relu" # other not work...
        self.semantic_attention_activation_fn = "sigmoid" # for now `None` not work

class LanguageModelConfig(object):
    def __init__(self):
        self.num_predicates = 70
        self.num_objects = 100
        self.dim_embedding = 300

        self.embedding_file = "./model/embeddings.npz"
        self.predicate_embedding_file = "./model/predicate_embeddings.npz"

        self.initializer_scale = 0.01

        #self.num_K_samples = 500000
        # K loss
        self.num_K_samples = 10000
        # self.coeff_K = 0.002
        self.coeff_K = 0.1

        # L loss
        self.num_L_samples = 5000
        self.coeff_L = 0.05
        # bigger temperature -> will choose less-frequent occured relationships as postive samples relatively frequently
        self.L_sample_temperature = 10.
        self.positive_relations_file = "./model/positive_relations.txt"

        # C loss
        self.visual_scores_file = "./visual_scores.txt"
        self.coeff_C = 1.

        self.min_pos_neg_diff = 5

        self.C_cross_image = True

        ## training configs
        self.initializer_scale = 0.01

        self.clip_gradients = 5.

        self.max_checkpoints_to_keep = 5

        self.optimizer = "Adam"

        self.batch_size = 32
