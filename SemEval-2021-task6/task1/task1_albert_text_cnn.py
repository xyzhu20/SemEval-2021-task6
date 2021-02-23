# -*- coding: utf-8 -*-
"""task1_albert_text_cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u1R0qLSdAYaNVQCRBojRNBVgg_RZHHsh
"""

# 2021-1-23 start test

# 挂载谷歌网盘
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
#
# !nvidia-smi

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/My Drive/SemEval-2021-task6(1-23)/task1"
# %ls

# !pip install transformers
# !pip install sentencepiece

# Commented out IPython magic to ensure Python compatibility.
# %ls

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer  # 多标签编码
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from load_data_task1 import load_train_data, load_dev_data, load_test_data  # 前面写的导入数据函数
from transformers import AlbertConfig, AlbertTokenizer, TFAlbertModel
import tensorflow as tf
import os
from metric import f1, Metrics

def save_result(dev_labels,dev_file,task_output_file):
    # 保存预测结果为txt文件形式
    import sys
    import json

    try:
        with open(dev_file, "r", encoding="utf8") as f:
            jsonobj = json.load(f)
    except:
        sys.exit("ERROR: cannot load json file")

    for i, example in enumerate(jsonobj):
        techniques_list = list(dev_labels[i])
        example['labels'] = techniques_list
        print("example %s: added %d labels" % (example['id'], len(techniques_list)))

    with open(task_output_file, "w") as fout:
        json.dump(jsonobj, fout, indent=4)
    print("Predictions written to file " + task_output_file)

class text_cnn(tf.keras.Model):
    def __init__(self):
        super(text_cnn, self).__init__()
        self.c1 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3), padding='VALID')  # 卷积层1
        self.b1 = tf.keras.layers.BatchNormalization()  # BN层1
        self.a1 = tf.keras.layers.Activation('relu')  # 激活层1
        self.p1 = tf.keras.layers.MaxPool1D(pool_size=(3), strides=2, padding='VALID')
        self.d1 = tf.keras.layers.Dropout(0.3)  # dropout层

        self.c2 = tf.keras.layers.Conv1D(filters=64, kernel_size=(4), padding='VALID')
        self.b2 = tf.keras.layers.BatchNormalization()  # BN层1
        self.a2 = tf.keras.layers.Activation('relu')  # 激活层1
        self.p2 = tf.keras.layers.MaxPool1D(pool_size=(4), strides=2, padding='VALID')
        self.d2 = tf.keras.layers.Dropout(0.3)  # dropout层

        self.c3 = tf.keras.layers.Conv1D(filters=64, kernel_size=(5), padding='VALID' )
        self.b3 = tf.keras.layers.BatchNormalization()  # BN层1
        self.a3 = tf.keras.layers.Activation('relu')  # 激活层1
        self.p3 = tf.keras.layers.MaxPool1D(pool_size=(5), strides=2, padding='VALID')
        self.d3 = tf.keras.layers.Dropout(0.3)  # dropout层

        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        x1 = self.c1(x)
        x1 = self.b1(x1)
        x1 = self.a1(x1)
        x1 = self.p1(x1)
        x1 = self.d1(x1)
        x1 = self.flatten(x1)

        x2 = self.c2(x)
        x2 = self.b2(x2)
        x2 = self.a2(x2)
        x2 = self.p2(x2)
        x2 = self.d2(x2)
        x2 = self.flatten(x2)

        x3 = self.c3(x)
        x3 = self.b3(x3)
        x3 = self.a3(x3)
        x3 = self.p3(x3)
        x3 = self.d3(x3)
        x3 = self.flatten(x3)

        y = tf.concat([x1,x2,x3],1)
        return y


def create_model():
    # bert层
    config = AlbertConfig.from_pretrained('albert-base-v2')
    print(config)
    bert_layer = TFAlbertModel.from_pretrained('albert-base-v2')
    initializer = tf.keras.initializers.TruncatedNormal(config.initializer_range)

    # 构建bert输入
    input_ids = tf.keras.Input(shape=(config.max_position_embeddings,), dtype='int32')
    token_type_ids = tf.keras.Input(shape=(config.max_position_embeddings,), dtype='int32')
    attention_mask = tf.keras.Input(shape=(config.max_position_embeddings,), dtype='int32')
    inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}

    # bert的输出
    bert_output = bert_layer(inputs)
    print(bert_output)
    hidden_states = bert_output[0]
    print(hidden_states)

    dropout_hidden = tf.keras.layers.Dropout(0.3)(hidden_states)
    text_cnn_layer = text_cnn()
    text_cnn_output = text_cnn_layer(dropout_hidden)
    dropout_output = tf.keras.layers.Dropout(0.3)(text_cnn_output)

    dense = tf.keras.layers.Dense(768, activation='relu')(dropout_output)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    output = tf.keras.layers.Dense(20, kernel_initializer=initializer, activation='sigmoid')(dropout)
    # output = output_layer(initializer)(dropout_output)
    # print(output)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    # print(config.max_position_embeddings)
    model.summary()
    return model

# 读取techiques列表
techiques_list_filename = '../techniques_list_task1-2.txt'
with open(techiques_list_filename, "r") as f:
    techiques_list = [line.rstrip() for line in f.readlines() if len(line) > 2]
print(techiques_list)

# load data
train_id, train_texts, train_techiques = load_train_data()
dev_id, dev_texts, dev_techniques = load_dev_data()
test_id, test_texts = load_test_data()

# 对标签进行编码
mlb = MultiLabelBinarizer(classes=techiques_list)
train_labels = mlb.fit_transform(train_techiques)
dev_labels = mlb.fit_transform(dev_techniques)



print("train data size:", len(train_texts))
print("val data size:", len(dev_texts))
print("test data size:", len(test_texts))

# 数据编码
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length')
dev_encodings = tokenizer(dev_texts, truncation=True, padding='max_length')
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length')

print(mlb)
print(techiques_list)
print(train_labels[15])
print(train_techiques[15])

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(dev_encodings),
    dev_labels
))

# 先用划分的验证集做测试以计算F1
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
))


# 创建网络
model = create_model()

# 断点续训
# checkpoint_save_path = "./checkpoint/albert_text_cnn_crossloss/muti_labels.ckpt"
model.load_weights("./checkpoint/albert_Crossentropy/muti_labels.ckpt") # best,0.625
checkpoint_save_path = "./checkpoint/albert_Crossentropy_more_epoch/muti_labels.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('===============Load Model=================')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                  monitor='val_f1',
                                                  mode='max', verbose=1,
                                                  save_weights_only=True,
                                                  save_best_only=True, )

tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./log/albert_Crossentropy_logs', profile_batch=0)

# 训练模型
optimizer = tf.keras.optimizers.Adam(lr=5e-6, epsilon=1e-8)
loss = tf.keras.losses.BinaryCrossentropy()  # 二进制交叉熵
metric = tf.keras.metrics.CategoricalAccuracy()
model.compile(optimizer=optimizer, loss=loss, metrics=[f1])

print("===============Start training=================")
history = model.fit(train_dataset.shuffle(1000).batch(8), validation_data=val_dataset.batch(8),epochs=300,
                        batch_size=8,
                        callbacks=[Metrics(valid_data=(val_dataset.batch(8), dev_labels)),
                                   cp_callback,
                                   tb_callback]
                        )


# 预测结果
#model = create_model()
# checkpoint_save_path = "./checkpoint/albert_text_cnn_crossloss/muti_labels.ckpt"
model.load_weights(checkpoint_save_path) # 取val最优模型
print("load_model")
y_pred = model.predict(val_dataset.batch(8))
y_true = dev_labels

print(len(y_pred),len(y_true))

# 求F1
import copy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
y_pred_t = copy.deepcopy(y_pred)
threshold = 0.5
upper, lower = 1, 0
y_pred_t[y_pred_t > threshold] = upper
y_pred_t[y_pred_t <= threshold] = lower
F1_Micro = f1_score(y_true, y_pred_t, average='micro')
F1_Macro = f1_score(y_true, y_pred_t, average='macro')
print("F1 macro:",F1_Macro)  
print("F1 MICRO",F1_Micro)

# !ls

# 在原先dev上预测
dev_file = "../development/dev_set_task1.txt"
def unlabel_dev_data():
  import sys
  import json
  try:
      with open(dev_file, "r", encoding='utf8') as f:
          jsonobj = json.load(f)
  except:
      sys.exit("ERROR: cannot load json file")

  id, text = [], []
  for example in jsonobj:
      id.append(example['id'])
      text.append(example['text'])

  # print("load dev data")
  return id, text
id, texts = unlabel_dev_data()
print(len(id),len(texts))
print(texts)

unlbale_dev_encodings = tokenizer(texts, truncation=True, padding='max_length')
unlbale_dev_dataset = tf.data.Dataset.from_tensor_slices((
    dict(unlbale_dev_encodings),
))

y_pred = model.predict(unlbale_dev_dataset.batch(8))
threshold, upper, lower = 0.5, 1, 0
y_pred[y_pred > threshold] = upper
y_pred[y_pred <= threshold] = lower

# 将结果转为labels list形式
unlbale_dev_labels = mlb.inverse_transform(y_pred)

# 保存预测结果
unlbale_dev_file = "../development/dev_set_task1.txt"
task_output_file = "../development/result/dev-task1-albert_Crossentropy.txt"
save_result(unlbale_dev_labels,unlbale_dev_file,task_output_file)

# !ls

# 在test上预测
y_pred = model.predict(test_dataset.batch(8))
threshold, upper, lower = 0.5, 1, 0
y_pred[y_pred > threshold] = upper
y_pred[y_pred <= threshold] = lower

# 将结果转为labels list形式
test_labels = mlb.inverse_transform(y_pred)

# 保存预测结果
test_file = "../data/test_set_task1.txt"
task_output_file = "./test-set-result/output-task1-albert_Crossentropy.txt"
save_result(test_labels,test_file,task_output_file)