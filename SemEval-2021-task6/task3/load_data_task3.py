'''load data for task3'''
test_file = "../data/test_set_task3/test_set_task3.txt"
training_file = "../data/training_set_task3/training_set_task3.txt"
dev_file = "../data/dev_set_task3_labeled/dev_set_task3_labeled.txt"

propaganda_techniques_file = "../techniques_list_task3.txt"

import sys
import json


# 读取标签列表
def load_techniques_list():
    with open(propaganda_techniques_file, "r") as f:
        propaganda_techniques_names = [line.rstrip() for line in f.readlines() if len(line) > 2]
    return propaganda_techniques_names


# 读取训练数据
def load_train_data():
    try:
        with open(training_file, "r", encoding="utf8") as f:
            jsonobj = json.load(f)
    except:
        sys.exit("ERROR: cannot load json file")

    id, text, labels, image = [], [], [], []
    for example in jsonobj:
        id.append(example['id'])
        text.append(example['text'])
        labels.append(example['labels'])
        image.append('../data/training_set_task3/' + example['image'])

    return id, text, labels, image

# 读取dev数据
def load_dev_data():
    try:
        with open(dev_file, "r", encoding="utf8") as f:
            jsonobj = json.load(f)
    except:
        sys.exit("ERROR: cannot load json file")

    id, text, labels, image = [], [], [], []
    for example in jsonobj:
        id.append(example['id'])
        text.append(example['text'])
        labels.append(example['labels'])
        image.append('../data/dev_set_task3_labeled/' + example['image'])

    return id, text, labels, image

# 读取test数据
def load_test_data():
    try:
        with open(test_file, "r", encoding="utf8") as f:
            jsonobj = json.load(f)
    except:
        sys.exit("ERROR: cannot load json file")

    id, text, image = [], [], []
    for example in jsonobj:
        id.append(example['id'])
        text.append(example['text'])
        image.append('../data/test_set_task3/' + example['image'])

    return id, text, image


if __name__ == '__main__':
    #dev_id, dev_texts, dev_labels, dev_image_path = load_dev_data()
    # print(dev_image_path)
    # print(dev_labels)
    # print("dev data size:", len(dev_texts))

    test_id ,test_text, test_image = load_test_data()
    print(test_image)
