import socket
import json
import sys
import time
import threading
import cv2
import torch
import numpy as np
from utils import combine_states
import torchvision
from resnet_utils import myResnet
from Model_strategy import Agent
from Batch import create_masks
import subprocess
from PyQt5.QtWidgets import QApplication
from PIL import Image, ImageQt
import os
import win32gui
import win32ui
import win32con
from utils import read_json
from utils import MyMNTDevice, 取图
from pynput.keyboard import Key, Listener
from pynput import keyboard
import random
from Model_strategy import Transformer
#window = int(subprocess.check_output(["xdotool", "search" ,"VehiclePhysicsExampleeeveed181"]).decode('ascii').split('\n')[0])
# _DEVICE_ID = '68UDU17B14011947'
_DEVICE_ID = '127.0.0.1:7555'  # mumu
# window = "RNE-AL00"
window = "R11"
window = win32gui.FindWindow(0, window)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
mod = torchvision.models.resnet101(pretrained=True).eval().cuda(device).requires_grad_(False)
resnet101 = myResnet(mod)
model_judge_state = Transformer(6, 768, 2, 12, 0.0, 6*6*2048)
model_judge_state.load_state_dict(torch.load('weights/model_weights_判断状态L'))
model_judge_state.cuda(device)
N = 15000  # 运行N次后学习
parallel = 100
episode = 3
lr = 0.0003
agent = Agent(act_num=7, parallel_num=parallel,
              lr=lr, episode=episode,
              input_size=6)


feedback_flag = True
total = 0
tmp_score = 0
tmp_score2 = 0


step = 0
learn_num = 0
allscores = []
allspeeds = []
bestscore = 0

time.sleep(1)
app = QApplication(sys.argv)
screen = app.primaryScreen()

# data_save_dir = '../training_data2'
time_start = 0


add_third_skill = 'd 0 559 1767 100\nc\nu 0\nc\n'
add_sec_skill = 'd 0 443 1562 100\nc\nu 0\nc\n'
add_fst_skill = 'd 0 246 1448 100\nc\nu 0\nc\n'
buy = 'd 0 636 190 100\nc\nu 0\nc\n'
# ope_com_dir="./json/ope_command.json"


# 词数词典路径 = "./json/词_数表.json"
# 数_词表路径 = "./json/数_词表.json"
# 操作查询路径 = "./json/名称_操作.json"
ope_dict = {"img_idx": "0", "move_ope": "无移动", "act_ope": "无动作"}

# if os.path.isfile(词数词典路径) and os.path.isfile(数_词表路径):
#     comb_idx, idx_comb = read_json(词数词典路径, 数_词表路径)
# with open(词数词典路径, encoding='utf8') as f:
#     词数词典 = json.load(f)
# with open(操作查询路径, encoding='utf8') as f:
#     ope_command_dict = json.load(f)


comb_idx_dir = "./json/comb_idx.json"
idx_comb_dir = "./json/idx_comb.json"
ope_com_dir = "./json/ope_command.json"

comb_idx = read_json(comb_idx_dir)
idx_comb = read_json(idx_comb_dir)
ope_command_dict = read_json(ope_com_dir)

direct_sheet = ['上移', '下移', '左移', '右移', '左上移', '左下移', '右上移', '右下移']
simulator = MyMNTDevice(_DEVICE_ID)
old_command = '移动停'

press1 = False
press2 = False
press3 = False
press4 = False
press5 = False
press6 = False
press7 = False
press8 = False
contflag = True
# 需要用一个东西来装关键事件
# 需要储存事件以及对应的图片

state_dict = {'击杀小兵或野怪或推掉塔': 1, '击杀敌方英雄': 5, '被击塔攻击': -2, '被击杀': -5, '无状况': 0, '死亡': 0, '其它': -0.03, '普通': 0}
# state_dict_A = {'击杀小兵或野怪或推掉塔': 0, '击杀敌方英雄': 1, '被击塔攻击': 2,  '被击杀': 3, '无状况': 4, '死亡': 5, '其它': 6, '普通': 7}

state_dict_B = {'击杀小兵或野怪或推掉塔': 0, '击杀敌方英雄': 1, '被击塔攻击': 2,  '被击杀': 3,  '死亡': 4, '普通': 5}

state = '无状况'
state_list = []
for K in state_dict_B:
    state_list.append(K)


def get_key_name(key):
    if isinstance(key, keyboard.KeyCode):
        return key.char
    else:
        return str(key)


def on_release(key):
    global press1, state
    key_name = get_key_name(key)
    if key_name == '1':
        press1 = False
    if key_name == '2':
        press2 = False
    if key_name == '3':
        press3 = False
    if key_name == '4':
        press4 = False
    if key_name == '5':
        press5 = False
    if key_name == '6':
        press6 = False
    if key_name == '7':
        press7 = False
    if key_name == '8':
        press8 = False
    if key_name == 'Key.page_down':
        state = '无状况'
    print("已经释放:", key_name)
    if key == Key.esc:
        # 停止监听
        return False


def on_press(key):
    global press1, state, contflag

    key_name = get_key_name(key)
    操作 = ''
    if key_name == 'Key.left':
        state = '击杀小兵或野怪或推掉塔'

    if key_name == 'Key.down':
        state = '击杀敌方英雄'
    if key_name == 'Key.right':
        state = '被击塔攻击'
    if key_name == 'Key.up':
        state = '被击杀'
    if key_name == 'Key.page_down':
        state = '其它'
    if key_name == 'q':
        state = '普通'
    if key_name == 'e':
        state = '死亡'
    if key_name == 'i':
        contflag = bool(1 - contflag)

    print(state)


def start_listen():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


th = threading.Thread(target=start_listen,)
th.start()


judge_data_dir = '../判断数据样本test'
if not os.path.exists(judge_data_dir):
    os.makedirs(judge_data_dir)

imgs_dir = judge_data_dir+'/'
if not os.path.exists(imgs_dir):
    os.mkdir(imgs_dir)

for i in range(6666666):
    img_tensor = torch.Tensor(0)
    ope_seq = np.ones((1, 1))
    ope_seq[0] = 128
    count = 0

    while contflag:
        time_start = time.time()

        img = screen.grabWindow(window)
        image = ImageQt.fromqimage(img)
        img_resize = image.resize((960, 480))
        #imgA = 取图(窗口名称)

        img_array = np.asarray(img_resize)
        tmp_tensor = torch.tensor(img_array, device=device)
        capture = tmp_tensor.unsqueeze(0).permute(0, 3, 2, 1) / 255
        _, out = resnet101(capture)
        out = torch.reshape(out, (1, 6*6*2048))

        ope_seqA = np.ones((1, 1))
        ope_tensorA = torch.tensor(ope_seqA.astype(np.int64)).cuda(device)

        src_mask, trg_mask = create_masks(ope_tensorA.unsqueeze(0), ope_tensorA.unsqueeze(0), device)
        outA = out.detach()

        real_output, _ = model_judge_state(outA.unsqueeze(0), ope_tensorA.unsqueeze(0), trg_mask)
        #实际输出=model_判断状态(out, 操作张量.unsqueeze(0),trg_mask)
        _, sample = torch.topk(real_output, k=1, dim=-1)
        sample_np = sample.cpu().numpy()

        if img_tensor.shape[0] == 0:
            img_tensor = out

        elif img_tensor.shape[0] < 120:
            img_tensor = torch.cat((img_tensor, out), 0)
            ope_seq = np.append(ope_seq, action)

        else:
            img_tensor = img_tensor[0:119, :]
            ope_seq = ope_seq[0:119]
            ope_seq = np.append(ope_seq, action)
            img_tensor = torch.cat((img_tensor, out), 0)

        ope_seqB = torch.tensor(ope_seq.astype(np.int64)).cuda(device)
        # ope_seqB = ope_seq.astype(np.)
        # src_mask, trg_mask = create_masks(ope_seq.unsqueeze(0), ope_seq.unsqueeze(0), device)
        src_mask, trg_mask = create_masks(ope_seqB.unsqueeze(0), ope_seqB.unsqueeze(0), device)

        cur_state = combine_states(img_tensor.cpu().numpy(), ope_seq, trg_mask)
        end = False
        episode_count = 0

        action, action_prob, critic = agent.select_action(cur_state, device, 0)

        # 周期性加一二三技能，并且停止移动
        if count % 50 == 0 and count != 0:
            simulator.发送(buy)
            simulator.发送(add_third_skill)
            simulator.发送(add_sec_skill)
            simulator.发送(add_fst_skill)
            simulator.发送(ope_command_dict['移动停'])
            print(old_command, '周期')
            time.sleep(0.02)
            simulator.发送(ope_command_dict[old_command])

        # 读取action 发送到设备
        command = idx_comb[str(action)]
        command_set = command.split('_')

        if command_set[0] == '无移动':
            command_set[0] = '移动停'

        if command_set[0] == old_command:
            ope_dict['move_ope'] = command_set[0]
            ope_dict['act_ope'] = command_set[1]

        else:
            ope_dict['move_ope'] = command_set[0]
            ope_dict['act_ope'] = command_set[1]
            old_command = command_set[0]
            simulator.发送(ope_command_dict[command_set[0]])

        time.sleep(0.01)
        if command_set[1] != '无动作' and command_set[1] != '发起集合' and command_set[1] != '发起进攻' and command_set[1] != '发起撤退':
            simulator.发送(ope_command_dict[command_set[1]])


#状态辞典={'击杀小兵或野怪或推掉塔': 1, '击杀敌方英雄': 5, '被击塔攻击': -2, '被击杀': -5,'无状况': 0, '死亡': 0, '其它': -0.03,'普通': 0}
        if state == '其它' or state == '无状况':
            state = state_list[sample_np[0, 0, 0, 0]]
        score = state_dict[state]

        # or 状况 == '被击塔攻击' or 状况 == '被击杀'
        #print(状况, '得分', 得分)
        
        # {'击杀小兵或野怪或推掉塔': 0, '击杀敌方英雄': 1, '被击塔攻击': 2,  '被击杀': 3,  '死亡': 4, '普通': 5}
        if state == '击杀小兵或野怪或推掉塔' or state == '击杀敌方英雄' or state == '被击塔攻击' or state == '被击杀':
            print(state, '得分', score)
            # 写入json
            if state == '击杀小兵或野怪或推掉塔' or state == '击杀敌方英雄':
                event_time = str(int(time.time()*100))

                img_resize.save(imgs_dir + event_time+'.jpg')
                event_dict = {}
                event_dict[event_time] = state
                rec_file = open(imgs_dir + '_判断数据.json', 'a+')
                json.dump(event_dict, rec_file, ensure_ascii=False)
                rec_file.write('\n')
                rec_file.close()
                
        elif state == '普通':
            if random.randint(0, 5000) % 100000 == 0: #TODO: 概率很小 只有random的结果为0时
                print(state, '得分', score)
                event_time = str(int(time.time()*100))

                img_resize.save(imgs_dir+ event_time+'.jpg')
                event_dict = {}
                event_dict[event_time] = state
                rec_file = open(imgs_dir + '_判断数据.json', 'a+')
                json.dump(event_dict, rec_file, ensure_ascii=False)
                rec_file.write('\n')
                rec_file.close()
                
        elif state == '死亡':
            if random.randint(0, 5000) % 50000 == 0: #TODO: 概率很小 只有random的结果为0时
                print(state, '得分', score)
                event_time = str(int(time.time()*100))

                img_resize.save(imgs_dir+event_time+'.jpg')
                event_dict = {}
                event_dict[event_time] = state
                rec_file = open(imgs_dir + '_判断数据.json', 'a+')
                json.dump(event_dict, rec_file, ensure_ascii=False)
                rec_file.write('\n')
                rec_file.close()
                
        if state != '其它':
            state = '无状况'
        else:
            print('其它得分', score)

        cur_state['img_tensor'] = cur_state['img_tensor'][:, -1:, :]
        cur_state['ope_seq'] = cur_state['ope_seq'][-1:]
        cur_state['trg_mask'] = 0
        #智能体.记录数据(状态, 动作, 动作可能性, 评价, 得分, 完结,计数)

        time_cost = 0.22 - (time.time() - time_start)
        if time_cost > 0:
            time.sleep(time_cost)

        count = count + 1
        if count % 10 == 0:
            print('time cost = {}'.format(time_cost))

        if contflag is False:

            print('learning.............')
            # 智能体.学习(device)
            print('score', 1)
            # 智能体.保存模型(学习次数)
            allscores = []
            allspeeds = []
            print('learning done')
            # 智能体.存硬盘('PPO训练数据/'+str(int(time.time())))
            # 智能体.保存模型(学习次数)

    time.sleep(1)
    print('继续', contflag)


#     状态=状态_
#     延迟 = 0.22 - (time.time() - 计时开始)
#     if 延迟 > 0:
#         time.sleep(延迟)
#     局内计数 = 局内计数 + 1
#
# 分数记录.append(分数)
#
# 平均分 = np.mean(分数记录[-500:])
# 平均速度 = np.mean(速度记录[-15000:])
# if 平均分 > 最高分:
#     最高分 = 平均分
#
# print('步数', 步数, '平均分', 平均分,'最高分',最高分,'局数',i,'平均速度',平均速度)

    # time.sleep(2)
    # while True:
    #
    #     time.sleep(11)
