from torch.autograd import Variable
import torch
import numpy as np
import json
import win32gui
import win32ui
import win32con
from pyminitouch import MNTDevice
from PyQt5.QtWidgets import QApplication
from PIL import Image, ImageQt
import sys


class MyMNTDevice(MNTDevice):
    def __init__(self, ID):
        MNTDevice.__init__(self, ID)

    def send_command(self, 内容):
        self.connection.send(内容)


def get_sample_data(idx_comb, data, score):
    tmp = data[0]
    list1 = [idx_comb[str(tmp[i, 0])] for i in range(0, tmp.shape[0])]
    tmp = score.cpu().numpy()
    list2 = [idx_comb[str(tmp[i])] for i in range(0, tmp.shape[0])]
    print("抽样输出", list1)
    print("目标输出", list2)


def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    np_mask = np_mask.cuda(device)
    return np_mask


def combine_states(img_tensor, ope_seq, trg_mask):
    state = {}
    state['img_tensor'] = img_tensor[np.newaxis, :]
    state['ope_seq'] = ope_seq
    state['trg_mask'] = trg_mask
    return state


def read_json(jsondir):
    with open(jsondir, encoding='utf-8') as f:
        results = json.load(f)
    return results


def capture_img(wname):
    hwnd = win32gui.FindWindow(0, wname)
    # hwnd = win32gui.FindWindow(0,'王者荣耀 - MuMu模拟器')
    app = QApplication(sys.argv)
    screen = QApplication.primaryScreen()
    img = screen.grabWindow(hwnd).toImage()
    image = ImageQt.fromqimage(img)
    # print(image.size)

    box = (0, 0, 960, 480)
    im2 = image.crop(box)

    return im2
