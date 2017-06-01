# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import numpy as np
import gensim
import cPickle
from collections import defaultdict

print("processing objects embeddings")
objects = json.load(open("objects.json", "r"))
word2vec = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
default_map = {
    "trash can": "bin",
    "traffic light": "light"
}
embeddings = np.concatenate([word2vec[default_map.get(obj, obj)][np.newaxis, :] for obj in objects], axis=0)
assert embeddings.shape == (100, 300)
embeddings.tofile("./model/embeddings.npz")

print("processing predicates embeddings")
processed_predicates = ["on", "wear", "has", "next", "sleep next", "sit next", "stand next", "park next", "walk next", "above", "behind", "stand behind", "sit behind", "park behind", "front", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "top", "left", "right", "sit on", "ride", "carry", "look", "stand on", "use", "at", "attach", "cover", "touch", "watch", "against", "inside", "adjacent", "across", "contain", "drive", "drive on", "taller", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", "face", "play with", "sleep on", "outside", "rest on", "follow", "hit", "feed", "kick", "skate on"]
predicate_embeddings = np.concatenate([np.mean([word2vec[word] for word in predicate.split() if word in word2vec], axis=0)[np.newaxis, :] for predicate in processed_predicates], axis=0)
assert predicate_embeddings.shape == (70, 300)
predicate_embeddings.tofile("./model/predicate_embeddings.npz")

# prepare positive relations
print("processing postive relations")
annotations = json.load(open("./annotations_train.json"))
occur_dct = defaultdict(lambda: 0)
relation_num_list = []
obj_num_list = []
for ann in annotations.itervalues():
    obj_set = set()
    for rel in ann:
        occur_dct[(rel["predicate"], rel["object"]["category"], rel["subject"]["category"])] += 1
        obj_set.add(rel["object"]["category"])
        obj_set.add(rel["subject"]["category"])
    relation_num_list.append(len(ann))
    obj_num_list.append(len(obj_set))
# TODO: 应该根据bbox判断是否为一个物体.. 不能只根据category
print("train集每张图片relation最多有 {} 个;\ntrain集每张图片object最多有 {} 个\n".format(np.max(relation_num_list),
                                                                                         np.max(obj_num_list)))

with open("./model/positive_relations.txt", "w") as f:
    f.write("\n".join(["{} {} {}\t{}".format(rel[0], rel[1], rel[2], occur) for rel, occur in occur_dct.iteritems()]))
