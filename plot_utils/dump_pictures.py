# -*- coding: utf-8 -*-
"""
Dump pictures for ablation study.
"""

import os
import json
import argparse

import PIL
from PIL import Image
from evaluate_utils import get_union_box

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_fname")
    parser.add_argument("dump_prefix")
    parser.add_argument("--annotation_file", default="annotations_train.json")
    parser.add_argument("--dataset_path", default="./sg_dataset/sg_train_images/")
    parser.add_argument("--preds", default=None)
    args = parser.parse_args()
    img = Image.open(os.path.join(args.dataset_path, args.img_fname))
    annotations = json.load(open(args.annotation_file, "r"))
    annotation = annotations[args.img_fname]

    for ann in annotation:
        image = img.crop(get_union_box(ann["object"]["bbox"], ann["subject"]["bbox"])).resize([224, 224], PIL.Image.BILINEAR)
        image.save("{}_{}_{}_{}.jpg".format(args.dump_prefix, ann["predicate"], ann["object"]["category"], ann["subject"]["category"]))

if __name__ == "__main__":
    main()
