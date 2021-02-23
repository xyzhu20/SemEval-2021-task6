# 处理并读取task2数据
import json
import pandas as pd
import copy
import pandas as pd
from transformers import AlbertTokenizerFast,BertTokenizerFast

#tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
#tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
def get_technique_list():
    # 读取techiques列表
    techiques_list_filename = '../techniques_list_task1-2.txt'
    with open(techiques_list_filename, "r", encoding='utf8') as f:
        techiques_list = [line.rstrip() for line in f.readlines() if len(line) > 2]
    return techiques_list

def BIO(text,fragment,technique,BIO_label):
    text = text.split()
    # 如果没有使用这项技术，返回全O
    if fragment == None:
        for i in range(len(BIO_label)):
            BIO_label[i] = 'O'
        return BIO_label

    fragment = fragment.split()
    # 判断开始
    for i in range(len(text)):
        if fragment[0] == text[i]:
            BIO_label[i] = 'B'
    # 判断结束
    if len(fragment)!=1:
        for i in range(len(text)):
            if fragment[-1] == text[i]:
                BIO_label[i] = 'I'
        # 将B和I中间填充为I
        for i in range(len(BIO_label)):
            if BIO_label[i] == 'B':
                for j in range(i+1,len(BIO_label)):
                    if BIO_label[j] == 'I':
                        break
                    else:
                        BIO_label[j] = 'I'
    # 添加技术名称
    # for i in range(len(BIO_label)):
    #   if BIO_label[i] == 'B' or BIO_label[i] == 'I' or BIO_label[i] == 'O':
    #       BIO_label[i] = BIO_label[i] + '-'+technique
    return BIO_label

# 读取train data
training_file = "./training_set_task2.txt"
def load_train_data():
    import sys
    import json
    try:
        with open(training_file, "r", encoding="utf8") as f:
            jsonobj = json.load(f)
    except:
        sys.exit("ERROR: cannot load json file")

    id, text, labels = [], [], []
    for example in jsonobj:
        id.append(example['id'])
        text.append(example['text'])
        labels.append(example['labels'])

    return id, text, labels

# def word_to_token(BIO_word,text):
#     # text为str形式
#     text_split = text.split()
#     # 处理好word形式的BIO后，转为token形式的BIO
#     BIO_token = []
#     for i in range(len(text_split)):
#         token = tokenizer.tokenize(text_split[i])
#         temp = [BIO_word[i]]*len(token)
#         BIO_token.extend(temp)
#     return BIO_token

if __name__ == '__main__':
    technique_list = get_technique_list()
    ids, texts, labels = load_train_data()
    #print(len(text_list))
    for i in range(len(ids)):
        df = pd.DataFrame()
        # 遍历每一条数据
        text = texts[i] # 将字母全部转为小写（使用的模型不区分大小写）
        text_ = text.split()

        #tokens = tokenizer.tokenize(text)
        #df['token'] = tokens
        df['text'] = text_
        # print(text_)
        # print(tokens)
        for technique in technique_list:
            # 对于每种技术，便利该条数据的labels字典,根据start和end找到fragment包含的单词
            # 并对照句子标注
            num = 0
            BIO_word = ['O']*len(text_)
            #print("original",BIO_label)
            for label in labels[i]:
                if technique == label['technique']:
                    fragment = text[label['start']:label['end']]
                    BIO_word = BIO(text,fragment,technique,BIO_word)
                    # print("技术",technique,end=' ')
                    # print("片段",fragment)
                    #print(BIO_label)
                    num = num + 1
            if num == 0:
                BIO_word = BIO(text,None,technique,BIO_word)
                # print("技术", technique, end=' ')
                # print("片段无")
                # print(BIO_label)
            print('word',BIO_word)
            # BIO_token = word_to_token(BIO_word,text)
            # print(tokens)
            # print('token',BIO_token)
            df[technique] = BIO_word
        path = './DataSet/BIO/'+str(ids[i])+'.csv'
        df.to_csv(path,index=False,encoding='utf-8')

