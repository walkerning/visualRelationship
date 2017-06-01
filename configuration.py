# -*- coding: utf-8 -*-

class ModelConfig(object):
    def __init__(self):
        ## model configs
        self.num_predicates = 700
        
        self.image_feature_name = "bbox"
        
        self.predicate_feature_name = "predicate"

        self.file_pattern = ""

        self.vgg_type = "vgg_16"

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
