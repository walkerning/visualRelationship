# -*- coding: utf-8 -*-

import json
import argparse
import cPickle
from pprint import pprint
import numpy as np
import inspect
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("save_file", help="The path to save the trained model")
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("load_file", help="The path to load the trained model")
    spec = inspect.getargspec(GBC.__init__.im_func)
    spec_args = spec.args[1:]
    deft = spec.defaults
    for ind, param_name in enumerate(spec_args):#GBC._get_param_names():
        if deft[ind] is None:
            tp = int
        else:
            tp = type(deft[ind])
        train_parser.add_argument("--" + param_name, type=tp, help="Set {} of GBC. type {}.".format(param_name, tp))

    train_ann_file = "./annotations_train.json"
    test_ann_file = "./annotations_test.json"
    args = parser.parse_args()
    print("Parsing the annotation files...")
    train_annotations = json.load(open(train_ann_file, "r"))
    test_annotations = json.load(open(test_ann_file, "r"))
    train_features, train_labels = parse_annotations(train_annotations)
    test_features, test_labels = parse_annotations(test_annotations)

    if args.phase == "train":
        print("Start fitting the classifier...")
        classifier = GBC()
        print("The hyper parameters of this GBC:")
        pprint(classifier.get_params())
        classifier.fit(train_features, train_labels)
    else:
        classifier = cPickle.load(open(args.load_file, "r"))
    train_predictions = classifier.predict(train_features)
    train_proba = classifier.predict_proba(train_features)
    eps = 1e-12
    train_cross_entropy_loss = -np.sum(np.log(proba[label] + eps) for proba, label in zip(train_proba, train_labels.astype(int)))
    print("train accuracy: {}".format(accuracy_score(train_labels, train_predictions)))
    print("train cross entropy loss: {}; mean: {}".format(train_cross_entropy_loss, train_cross_entropy_loss / len(train_labels)))
    test_proba = classifier.predict_proba(test_features)
    test_cross_entropy_loss = -np.sum(np.log(proba[label] + eps) for proba, label in zip(test_proba, test_labels.astype(int)))
    print("test accuracy: {}".format(classifier.score(test_features, test_labels)))
    print("test cross entropy loss: {}; mean: {}".format(test_cross_entropy_loss, test_cross_entropy_loss / len(test_labels)))
    if args.phase == "train":
        print("Saving model to {}.".format(args.save_file))
        cPickle.dump(classifier, open(args.save_file, "w"))
    
def parse_annotations(annotations):
    total_legal_rel = np.sum([len(anns) for anns in annotations.itervalues()])
    features = np.zeros((total_legal_rel, 10))
    labels = np.zeros((total_legal_rel,))
    index = 0
    for _, anns in annotations.iteritems():
        for ann in anns:
            features[index] = [ann["object"]["category"], ann["subject"]["category"]] + ann["object"]["bbox"] + ann["subject"]["bbox"]
            labels[index] = ann["predicate"]
            index += 1
    assert index == total_legal_rel
    return features, labels.astype(int)

if __name__ == "__main__":
    main()
