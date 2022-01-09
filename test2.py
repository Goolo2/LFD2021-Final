import torch


# 改weight的key名

wdict = torch.load('./weights/state_model_weight')
wdict['actions.weight'] = wdict.pop('outX.weight')
wdict['actions.bias'] = wdict.pop('outX.bias')
wdict['critics.weight'] = wdict.pop('评价.weight')
wdict['critics.bias'] = wdict.pop('评价.bias')
wdict['graph_trans.weight'] = wdict.pop('图转.weight')
wdict['graph_trans.bias'] = wdict.pop('图转.bias')


torch.save(wdict, 'weights/state_model_weight_new')


# 筛选掉无用的图片
import json
import os

root = r'E:\TBSI\课程\数据学习\\final\NEWADB\WZCQ-main\dataset\unused'

for fo in os.listdir(root):
    print(f'folder {fo}')

    fopath = os.path.join(root, fo)
    jsonpath = os.path.join(root, fo, 'operations.json')

    # with open(jsonpath, 'r', encoding='utf-8') as f:
    #     data = json.load(f)

    record_file = open(os.path.join(root, fo, 'newoperations.json'), 'w+')

    data = []
    with open(jsonpath, encoding='ansi') as f:
        while True:
            df = f.readline()
            df = df.replace('\'', '\"')

            if df == "":
                break
            df = json.loads(df)
            data.append(df)

    imgset = set()
    for item in os.listdir(fopath):
        if item.split('.')[-1] == 'jpg':
            imgset.add(item.split('.')[0])

    newdata = []
    for line in data:
        if line['img_idx'] in imgset:
            newdata.append(line)
            json.dump(line, record_file, ensure_ascii=False)
            record_file.write('\n')

    # newpath = os.path.join(root, fo, 'newoperations.json')
    # with open(newpath, 'w') as f:
    #     json.dump(newdata, f, indent=2, ensure_ascii=False)

    record_file.close()

    print(len(data))
    print(len(newdata))
