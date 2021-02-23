import sys
import json
import random

dev_file = "../data/dev_set_task1.txt"
train_file = "../data/training_set_task1.txt"
test_file = "../data/test_set_task1.txt"

def load_test_data():
    try:
        with open(test_file, "r", encoding='utf8') as f:
            jsonobj = json.load(f)
    except:
        sys.exit("ERROR: cannot load json file")

    id, text = [], []
    for example in jsonobj:
        id.append(example['id'])
        text.append(example['text'])

    # print("load dev data")
    return id, text


def load_train_data():
    try:
        with open(train_file, "r", encoding='utf8') as f:
            jsonobj = json.load(f)
    except:
        sys.exit("ERROR: cannot load json file")

    id, text, labels = [], [], []
    for example in jsonobj:
        id.append(example['id'])
        text.append(example['text'])
        labels.append(example['labels'])

    # print("load train data")
    return id, text, labels

def load_dev_data():
    try:
        with open(dev_file, "r", encoding='utf8') as f:
            jsonobj = json.load(f)
    except:
        sys.exit("ERROR: cannot load json file")

    id, text, labels = [], [], []
    for example in jsonobj:
        id.append(example['id'])
        text.append(example['text'])
        labels.append(example['labels'])

    # print("load train data")
    return id, text, labels

if __name__ == '__main__':
    load_train_data()
