'''
@File    :   check_json.py
@Time    :   2021/12/29 22:12:54
@Author  :   goole 
@Version :   1.0
@Discrib :   检查mintouch的指令控制
'''

import json
from utils import *
import os
import time


_DEVICE_ID = '127.0.0.1:7555'  # mumu
设备 = MyMNTDevice(_DEVICE_ID)

# Json file check
# with open('./json/词_数表.json', 'rb') as f:
#     data = json.load(f)
#     print(data)
# with open('./json/数_词表.json', 'rb') as f:
#     data = json.load(f)
#     print(data)
# with open('./json/名称_编号.json', 'rb') as f:
#     data = json.load(f)
#     print(data)


# data = {
#     # "攻击": "d 0 169 1900 100\nc\nu 0\nc\n",
#     # "补刀": "d 0 95 1700 100\nc\nu 0\nc\n",
#     # "推塔": "d 0 310 1950 100\nc\nu 0\nc\n",
#     "一技能": "d 0 133 1660 100\nc\nu 0\nc\n",
#     # "二技能": "d 0 342 1782 100\nc\nu 0\nc\n",
#     # "三技能": "d 0 455 1984 100\nc\nu 0\nc\n",
#     # "召唤师技能": "d 0 117 1496 100\nc\nu 0\nc\n",
#     # "回城": "d 0 108 1206 100\nc\nu 0\nc\n",
#     # "发起进攻": "d 0 945 2110 100\nc\nu 0\nc\n",
#     # "发起撤退": "d 0 851 2112 100\nc\nu 0\nc\n",
#     # "发起集合": "d 0 765 2110 100\nc\nu 0\nc\n",
#     # "上移": "d 1 237 321 300\nc\nm 1 349 321 100\nc\n",
#     # "右移": "d 1 237 321 300\nc\nm 1 237 434 100\nc\n",
#     # "下移": "d 1 237 321 300\nc\nm 1 180 321 100\nc\n",
#     # "左移": "d 1 237 321 300\nc\nm 1 237 209 100\nc\n",
#     # "左上移": "d 1 237 321 300\nc\nm 1 315 243 100\nc\n",
#     # "左下移": "d 1 237 321 300\nc\nm 1 158 243 100\nc\n",
#     # "右下移": "d 1 237 321 300\nc\nm 1 158 400 100\nc\n",
#     # "右上移": "d 1 237 321 300\nc\nm 1 315 400 100\nc\n",
#     # "移动停": "u 1\nc\n",
#     # "恢复": "d 0 111 1345 100\nc\nu 0\nc\n"
# }
data = {
    # "攻击": "d 0 151 1864 100\nc\nu 0\nc\n",
    # "补刀": "d 0 95 1700 100\nc\nu 0\nc\n",
    # "推塔": "d 0 313 1954 100\nc\nu 0\nc\n",
    # "一技能": "d 0 132 1548 100\nc\nu 0\nc\n",
    # "二技能": "d 0 330 1666 100\nc\nu 0\nc\n",
    # "三技能": "d 0 452 1871 100\nc\nu 0\nc\n",
    # "召唤师技能": "d 0 450 1863 100\nc\nu 0\nc\n",
    # "回城": "d 0 112 1082 100\nc\nu 0\nc\n",
    # "发起进攻": "d 0 931 2105 100\nc\nu 0\nc\n",
    # "发起撤退": "d 0 842 2102 100\nc\nu 0\nc\n",
    # "发起集合": "d 0 752 2100 100\nc\nu 0\nc\n",
    # "上移": "d 1 237 321 300\nc\nm 1 401 435 100\nc\n",
    # "右移": "d 1 237 321 300\nc\nm 1 234 602 100\nc\n",
    # "下移": "d 1 237 321 300\nc\nm 1 68 438 100\nc\n",
    # "左移": "d 1 237 321 300\nc\nm 1 236 273 100\nc\n",
    # "左上移": "d 1 237 321 300\nc\nm 1 366 319 100\nc\n",
    # "左下移": "d 1 237 321 300\nc\nm 1 121 300 100\nc\n",
    # "右下移": "d 1 237 321 300\nc\nm 1 112 567 100\nc\n",
    # "右上移": "d 1 237 321 300\nc\nm 1 341 579 100\nc\n",
    # "移动停": "u 1\nc\n",
    # "恢复": "d 0 108 1232 100\nc\nu 0\nc\n",
    
    '加三技能':'d 0 559 1767 100\nc\nu 0\nc\n',
    '加二技能':'d 0 443 1562 100\nc\nu 0\nc\n',
    '加一技能':'d 0 246 1448 100\nc\nu 0\nc\n',
    '购买':'d 0 636 190 100\nc\nu 0\nc\n',
}

for k,v in data.items():
    print(k,v)
    for i in range(8):
        设备.发送(v)
        time.sleep(1)

# with open('./json/名称_操作.json', 'rb') as f:
#     data = json.load(f)
#     for key, value in data.items():
#         print(key, value)
#         for i in range(10):
#             设备.发送(value)
#             time.sleep(1)

#         print('==========================================')
#         myinput = input('c')
#         if myinput != 'c':
#             break
    # print(data)


# On-sereen Pointer Visualization
# _DEVICE_ID = 'db5fece5'
# 窗口名称="MIX 2S"
# 设备 = MyMNTDevice(_DEVICE_ID)
# 设备.发送('d 0 1860 930 100\nc\nu 0\nc\n')
# print('Button pressed.')
# while True:
#     设备.发送('d 1 1000 2100 100\nc\nu 0\nc\n')
#     # 设备.发送('d 0 1860 930 100\nc\nu 0\nc\n')
#     time.sleep(1)


# Establish a local json file for the local cell phone layout in game
# d <contact> <x> <y> <pressure>
# m <contact> <x> <y> <pressure>
# see more touch commands in minitouch repo: https://github.com/openstf/minitouch
dict_layout = {
    # coordination in Android adb: (X,Y)
    # '攻击': 'd 0 1860 930 100\nc\nu 0\nc\n',
    # '补刀': 'd 0 1687 988 100\nc\nu 0\nc\n',
    # '推塔': 'd 0 1950 775 100\nc\nu 0\nc\n',
    #
    # '一技能': 'd 0 1532 952 100\nc\nu 0\nc\n',
    # '二技能': 'd 0 1673 761 100\nc\nu 0\nc\n',
    # '三技能': 'd 0 1857 645 100\nc\nu 0\nc\n',
    # '召唤师技能': 'd 0 1373 965 100\nc\nu 0\nc\n',
    #
    # '回城': 'd 0 1090 962 100\nc\nu 0\nc\n',
    #
    # '发起进攻': 'd 0 2099 155 100\nc\nu 0\nc\n',
    # '发起撤退': 'd 0 2108 245 100\nc\nu 0\nc\n',
    # '发起集合': 'd 0 2100 338 100\nc\nu 0\nc\n',
    #
    # '上移': 'd 1 430 840 300\nc\nm 1 430 710 100\nc\n',  # controller center -> moving end point
    # '右移': 'd 1 430 840 300\nc\nm 1 560 840 100\nc\n',
    # '下移': 'd 1 430 840 300\nc\nm 1 430 970 100\nc\n',
    # '左移': 'd 1 430 840 300\nc\nm 1 300 840 100\nc\n',
    # '左上移': 'd 1 430 840 300\nc\nm 1 330 740 100\nc\n',
    # '左下移': 'd 1 430 840 300\nc\nm 1 330 940 100\nc\n',
    # '右下移': 'd 1 430 840 300\nc\nm 1 530 940 100\nc\n',
    # '右上移': 'd 1 430 840 300\nc\nm 1 530 740 100\nc\n',
    #
    # '移动停': 'u 1\nc\n',
    # '恢复': 'd 0 1225 985 100\nc\nu 0\nc\n',
    #############################################
    # coordination in minitouch (X,Y) -> (1080-Y,X)
    '攻击': 'd 0 150 1860 100\nc\nu 0\nc\n',  # 930(150)  Y(1080-Y, i.e. new 'X')
    '补刀': 'd 0 92 1687 100\nc\nu 0\nc\n',  # 988(92)
    '推塔': 'd 0 305 1950 100\nc\nu 0\nc\n',  # 775(305)

    '一技能': 'd 0 128 1532 100\nc\nu 0\nc\n',  # 952(128)
    '二技能': 'd 0 319 1673 100\nc\nu 0\nc\n',  # 761(319)
    '三技能': 'd 0 435 1857 100\nc\nu 0\nc\n',  # 645(435)
    '召唤师技能': 'd 0 115 1373 100\nc\nu 0\nc\n',  # 965(115)

    '回城': 'd 0 118 1090 100\nc\nu 0\nc\n',  # 962(118)

    '发起进攻': 'd 0 925 2099 100\nc\nu 0\nc\n',  # 155(925)
    '发起撤退': 'd 0 835 2108 100\nc\nu 0\nc\n',  # 245(835)
    '发起集合': 'd 0 742 2100 100\nc\nu 0\nc\n',  # 338(742)

    '上移': 'd 1 240 430 300\nc\nm 1 370 430 100\nc\n',  # controller center -> moving end point  # 840(240) 710(370)
    '右移': 'd 1 240 430 300\nc\nm 1 240 560 100\nc\n',  # 840 840(240)
    '下移': 'd 1 240 430 300\nc\nm 1 110 430 100\nc\n',  # 840 970(110)
    '左移': 'd 1 240 430 300\nc\nm 1 240 300 100\nc\n',  # 840 840(240)
    '左上移': 'd 1 240 430 300\nc\nm 1 340 330 100\nc\n',  # 840 740(340)
    '左下移': 'd 1 240 430 300\nc\nm 1 140 330 100\nc\n',  # 840 940(140)
    '右下移': 'd 1 240 430 300\nc\nm 1 140 530 100\nc\n',  # 840 940(140)
    '右上移': 'd 1 240 430 300\nc\nm 1 340 530 100\nc\n',  # 840 740(340)

    '移动停': 'u 1\nc\n',  # 930
    '恢复': 'd 0 1225 985 100\nc\nu 0\nc\n'  # 930
}

# test button press
# while True:
#     设备.发送('d 1 240 430 100\nc\nu 0\nc\n')
#     # 设备.发送('d 1 240 430 300\nc\nm 1 370 430 100\nc\n')
#     time.sleep(1)

# with open('./json/local_layout.json', 'w') as json_file:
#     json.dump(dict_layout, json_file)

# with open('./json/local_layout.json', 'rb') as f:
#     data = json.load(f)
#     print(data)
