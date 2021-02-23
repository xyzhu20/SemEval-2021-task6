import pandas as pd
import copy
def get_technique_list():
    # 读取techiques列表
    techiques_list_filename = '../techniques_list_task1-2.txt'
    with open(techiques_list_filename, "r", encoding='utf8') as f:
        techiques_list = [line.rstrip() for line in f.readlines() if len(line) > 2]
    return techiques_list

# 读取test data
training_file = "../data/test_set_task2.txt"
def load_test_data():
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
        #labels.append(example['labels'])

    return id, text

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

if __name__ == '__main__':
    technique_list = get_technique_list()
    ids, texts = load_test_data()

    result = []
    for id in range(len(ids)):
        # 对于每一个句子
        path = './DataSet/test_result/' + str(ids[id]) + '.csv'
        #path = './DataSet/dev_BIO/' + '112_batch_2' + '.csv'
        df = pd.read_csv(path,header = 0)
        label = []
        state = 0
        for tech in technique_list:
            temp = dict()
            tag = 'I-' + tech
            #print(tag)
            if tag in df[tech].values:
                state = 1
                print("tech:",tech)
                # 找开头
                s,e = 0,0
                for i in range(len(df[tech])):
                    if df[tech][i] == tag:
                        start = df['text'][i]
                        print("start:",start)
                        s = texts[id].find(start)
                        break
                # 找结尾
                for j in range(len(df[tech])-1,-1,-1):
                    if df[tech][j] == tag:
                        end = df['text'][j]
                        print("end:",end)
                        e = texts[id].find(end)
                        t = copy.deepcopy(e)
                        while t < len(texts[id]):
                            t = t + 1
                            if texts[e:t] == end:
                                break
                        e = t
                        break
                temp["start"] = s
                temp["end"] = e
                temp["technique"] = tech
                temp["text_fragment"] = texts[id][s:e]
                print(temp)
                print("Save Complete")
                label.append(temp)
        result.append(label)

    output_file = "./task2-test_set-output.txt"
    output_labels = result
    dev_file = "../data/test_set_task2.txt" # test数据地址
    save_result(output_labels,dev_file,output_file)



