import torch
import torchvision
from PIL import Image
import numpy as np
import time
import json
from config import GPT2Config, TransformerConfig
from Batch import create_masks

import torch.nn.functional as F
from utils import *
import os
import random
from Model_strategy import Transformer
from Model_strategy import Agent


state_dict_B = {'击杀小兵或野怪或推掉塔': 0, '击杀敌方英雄': 1, '被击塔攻击': 2,  '被击杀': 3,  '死亡': 4, '普通': 5}
state_dict = {'击杀小兵或野怪或推掉塔': 2, '击杀敌方英雄': 5, '被击塔攻击': -0.5, '被击杀': -2, '无状况': 0.01, '死亡': 0.01, '其它': -0.003, '普通': 0.01}
state_list = []
for K in state_dict_B:
    state_list.append(K)

datadir = '../dataset/unused'
if not os.path.exists(datadir):
    os.makedirs(datadir)
for root, dirs, files in os.walk('../dataset/unused'):
    if len(dirs) > 0:
        break

comb_idx_dir = "./json/comb_idx.json"
idx_comb_dir = "./json/idx_comb.json"

comb_idx = read_json(comb_idx_dir)
idx_comb = read_json(idx_comb_dir)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
config = TransformerConfig()
# 模型路径 = 'model_weights_2021-05-7D'

model_judge_state = Transformer(6, 768, 2, 12, 0.0, 6*6*2048)
# model_judge_state.load_state_dict(torch.load('weights/model_weights_判断状态L'))
model_judge_state.load_state_dict(torch.load('weights/state_model_weight_new'))
model_judge_state.cuda(device).requires_grad_(False)

N = 15000  # 运行N次后学习
parallel = 100
episode = 3
lr = 0.0003
agent = Agent(act_num=7, parallel_num=parallel,
              lr=lr, episode=episode,
              input_size=6)


chunksize = 600
idxsize = 600
branch = 1

count = 0
time_start = time.time()
for j in range(100):
    # for j in range(1):
    print(f'epoch = {j}')
    # random.shuffle(dirs)
    for folder in dirs:
        allrewards = []
        predata = 'E:/TBSI/课程/数据学习/final/NEWADB/WZCQ-main/dataset/unused/' + folder + '/processed_data.npz'
        if os.path.isfile(predata):
            npzdata = np.load(predata, allow_pickle=True)
            img_tensornp, ope_seq = npzdata["img_tensornp"], npzdata["ope_seq"]
            if img_tensornp.shape[0] < 600:
                continue
            loop = True
            idx = 0
            # TODO:为啥要插入128？？？？
            ope_seq = np.insert(ope_seq, 0, 128)

            ope_score_table = []
            tgtoutput_score_table = []
            pic_score_table = []

            while loop:
                if idx + chunksize < ope_seq.shape[0]:
                    ope_score = ope_seq[idx:idx + chunksize]
                    tgtoutput_score = ope_seq[idx + 1:idx + 1 + chunksize]
                    pic_score = img_tensornp[idx:idx + chunksize, :]

                    ope_score_table.append(ope_score)
                    tgtoutput_score_table.append(tgtoutput_score)
                    pic_score_table.append(pic_score)
                    idx = idx + idxsize
                else:
                    ope_score = ope_seq[-chunksize - 1:-1]
                    tgtoutput_score = ope_seq[-chunksize:]
                    pic_score = img_tensornp[-chunksize:, :]

                    ope_score_table.append(ope_score)
                    tgtoutput_score_table.append(tgtoutput_score)
                    pic_score_table.append(pic_score)
                    loop = False

            loop = True
            i = 0
            while loop:
                if (i+1)*branch < len(ope_score_table):
                    ope_branch = np.array(ope_score_table[i * branch: (i+1) * branch])
                    pic_branch = np.array(pic_score_table[i * branch: (i + 1) * branch])
                    tgtoutput_branch = np.array(tgtoutput_score_table[i * branch: (i + 1) * branch])

                else:
                    ope_branch = np.array(ope_score_table[i * branch: len(ope_score_table)])
                    pic_branch = np.array(pic_score_table[i * branch: len(pic_score_table)], dtype=np.float32)
                    tgtoutput_branch = np.array(tgtoutput_score_table[i * branch: len(tgtoutput_score_table)])
                    loop = False

                ope_score_tensor = torch.tensor(ope_branch).cuda(device)
                ope_seqA = np.ones_like(ope_branch)
                ope_seqA_tensor = torch.tensor(ope_seqA).cuda(device)
                pic_score_tensor = torch.tensor(pic_branch).cuda(device)
                tgtoutput_score_tensor = torch.tensor(tgtoutput_branch).cuda(device)

                src_mask, trg_mask = create_masks(ope_score_tensor, ope_score_tensor, device)  # TODO:??
                if pic_score_tensor.shape[0] != ope_score_tensor.shape[0]:
                    continue

                state = {}
                state['ope_seq'] = ope_branch
                state['img_tensor'] = pic_branch
                state['trg_mask'] = trg_mask

                # TODO: 这里开了手动的flag可能有问题，
                action, action_prob, critic = agent.select_action_batch(state, device, tgtoutput_score_tensor, True)

                real_output, _ = model_judge_state(pic_score_tensor, ope_seqA_tensor, trg_mask)

                _, sample = torch.topk(real_output, k=1, dim=-1)
                samplenp = sample.cpu().numpy()
                reward = np.ones_like(samplenp[0, :, 0])
                reward = reward.astype(np.float32)
                for count in range(samplenp.shape[1]):
                    sate = state_list[samplenp[0, count, 0]]

                    score = state_dict[sate]
                    reward[count] = score

                agent.supervised_rl(device, state, reward, action, action_prob, critic)
                mean_reward = np.mean(reward)
                print('mean reward = {}'.format(mean_reward))
                # 输出_实际_A = model(图片_分_torch,操作_分_torch ,trg_mask)
                # lin = 输出_实际_A.view(-1, 输出_实际_A.size(-1))
                # optimizer.zero_grad()
               # loss = F.cross_entropy(lin, 目标输出_分_torch.contiguous().view(-1), ignore_index=-1)
                if count % 1 == 0:
                    # print(loss)

                    time_end = time.time()
                    duringtime = time_end - time_start

                    #_, 抽样 = torch.topk(输出_实际_A, k=1, dim=-1)
                    #抽样np = 抽样.cpu().numpy()
                    #打印抽样数据(idx_comb, 抽样np[0:1,:,:], 目标输出_分_torch[0,:])
                    print("time{} epoch{} count{} folder{}".format(duringtime, j, count, folder))

                if count % 45060 == 0:
                    print('888')

                # loss.backward()
                # optimizer.step()
                count = count+1
                i = i+1
agent.savemodel('0109')
