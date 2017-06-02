# -*- coding: utf-8 -*-
"""
This is a deprecated script for preparing inital pretrained vgg checkpoint.
As I think there is no available pretrained VGG16 model in tensorflow's format...

Now, use this checkpointdirectly: https://github.com/tensorflow/models/tree/master/slim#Pretrained
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
params = np.load(open("./model/data", "r")).tolist()
assert isinstance(params, dict)
from configuration import ModelConfig
from visual_relation_module import VisualModule

model_config = ModelConfig()
model_config.file_pattern = "./*"
model = VisualModule(model_config, "train")
model.build()
assigns = []
for x in model.conv_variables:
    assert "conv" in x.op.name
    # if "conv" not in x.op.name:
    #     # only initialize convolution layers
    #     continue
    # ignore scopes
    name, role = x.op.name.split("/")[-2:]
    assert name in params
    print("assigning `{}` `{}`".format(name, role))
    assigns.append(x.assign(params[name][role]))

whole_assign = tf.group(*assigns)

saver = tf.train.Saver(model.conv_variables)
with tf.Session() as sess:
    sess.run(whole_assign)
    saver.save(sess, "model/initial_vgg_ckpt")

# from tensorflow.python.tools import inspect_checkpoint
# saved_tensors = inspect_checkpoint.print_tensors_in_checkpoint_file("./model/initial_vgg_ckpt", [], True)
reader = pywrap_tensorflow.NewCheckpointReader("model/initial_vgg_ckpt")
var_to_shape_map = reader.get_variable_to_shape_map()
print("tensor_names: ", var_to_shape_map.keys())
