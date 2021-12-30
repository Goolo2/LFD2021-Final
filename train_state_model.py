from random import shuffle
import torch
import torchvision
import json
from PIL import Image
from resnet_utils import myResnet
import numpy as np
import torch.nn as nn
from Sublayers import Norm, MLP
import math
import torch.nn.functional as F
from Model_strategy import Transformer
from Batch import create_masks
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
resnet101 = torchvision.models.resnet101(pretrained=True).eval()
resnet101 = myResnet(resnet101).cuda(device).requires_grad_(False)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Judge_state(nn.Module):
    def __init__(self, output_size, hidden_size, input_size=2048, input_sizeA=36):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.input_sizeA = input_sizeA
        self.input_layer = MLP(input_size, hidden_size)
        self.hidden_layer = MLP(hidden_size, hidden_size)
        self.output_layer = MLP(hidden_size*input_sizeA, output_size)

    def forward(self, graph_vec):
        graph_vec = graph_vec.reshape((graph_vec.shape[0], self.input_sizeA, self.input_size))
        tmp_vec = gelu(self.input_layer(graph_vec))
        tmp_vec = self.hidden_layer(tmp_vec)
        tmp_vec = tmp_vec.reshape((tmp_vec.shape[0], self.hidden_size*self.input_sizeA))
        results = self.output_layer(tmp_vec)
        return results


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic


# model_判断状态=判断状态(6,1024,2048).cuda(device)
model_judge_state = Transformer(6, 768, 2, 12, 0.0, 6*6*2048).cuda(device)
# model_判断状态.load_state_dict(torch.load('weights/model_weights_判断状态C1'))
optimizer = torch.optim.Adam(model_judge_state.parameters(), lr=6.25e-5, betas=(0.9, 0.98), eps=1e-9)
jsonpath = '../判断数据样本/判断新.json'


alldata = {}
state_dict = {'击杀小兵或野怪或推掉塔': 0, '击杀敌方英雄': 1, '被击塔攻击': 2,  '被击杀': 3,  '死亡': 4, '普通': 5}
state_list = []

for K in state_dict:
    state_list.append(K)

with open(jsonpath, 'w', encoding='ansi') as f:
    while True:
        df = f.readline()
        df = df.replace('\'', '\"')
        if df == "":
            break
        data = json.loads(df)
        for key in data:
            alldata[key] = data[key]

state = np.ones((1, ), dtype='int64')
for i in range(100):
    shuffle_order = random_dic(alldata)

    for key in shuffle_order:
        state_num = state_dict[alldata[key]]

        state[0] = state_num
        target_output = torch.from_numpy(state).cuda(device)

        img_dir = '../判断数据样本/' + key + '.jpg'
        img = Image.open(img_dir)
        img2 = np.array(img)

        img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1).float() / 255
        _, out = resnet101(img2)
        img_tensor = out.reshape(1, 6 * 6 * 2048)
        ope_seq = np.ones((1, 1))
        ope_tensor = torch.from_numpy(ope_seq.astype(np.int64)).cuda(device)
        src_mask, trg_mask = create_masks(ope_tensor.unsqueeze(0), ope_tensor.unsqueeze(0), device)
        real_output, _ = model_judge_state(img_tensor.unsqueeze(0), ope_tensor.unsqueeze(0), trg_mask)

        _, sample = torch.topk(real_output, k=1, dim=-1)
        sample_np = sample.cpu().numpy()

        optimizer.zero_grad()
        real_output = real_output.view(-1, real_output.size(-1))
        loss = F.cross_entropy(real_output, target_output.contiguous().view(-1), ignore_index=-1)
        print('轮', i, '实际输出', state_list[sample_np[0, 0, 0, 0]], '目标输出', alldata[key], loss)
        loss.backward()

        optimizer.step()

    # torch.save(model_判断状态.state_dict(), 'weights/model_weights_判断状态L')

    # torch.save(model_判断状态.state_dict(), 'weights/model_weights_判断状态L{}'.format(str(i)))
    torch.save(model_judge_state.state_dict(), 'weights/model_weights_newcxc{}'.format(str(i)))
