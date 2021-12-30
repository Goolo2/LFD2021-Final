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


# def 取图(窗口名称):
#     # 获取后台窗口的句柄，注意后台窗口不能最小化
#     hWnd = win32gui.FindWindow(0,窗口名称)  # 窗口的类名可以用Visual Studio的SPY++工具获取
#     # 获取句柄窗口的大小信息
#     left, top, right, bot = win32gui.GetWindowRect(hWnd)
#     width = right - left
#     height = bot - top
#     # 返回句柄窗口的设备环境，覆盖整个窗口，包括非客户区，标题栏，菜单，边框
#     hWndDC = win32gui.GetWindowDC(hWnd)
#     # 创建设备描述表
#     mfcDC = win32ui.CreateDCFromHandle(hWndDC)
#     # 创建内存设备描述表
#     saveDC = mfcDC.CreateCompatibleDC()
#     # 创建位图对象准备保存图片
#     saveBitMap = win32ui.CreateBitmap()
#     # 为bitmap开辟存储空间
#     saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
#     # 将截图保存到saveBitMap中
#     saveDC.SelectObject(saveBitMap)
#     # 保存bitmap到内存设备描述表
#     saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)


#     bmpinfo = saveBitMap.GetInfo()
#     bmpstr = saveBitMap.GetBitmapBits(True)
#     ###生成图像
#     im_PIL = Image.frombuffer('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpstr,'raw','BGRX')
#     #im_PIL= Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr)
#     #im_PIL =Image.frombytes('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpstr)
#     box = (8,31,968,511)
#     im2 = im_PIL.crop(box)
#     #im2.save('./dd2d.jpg')
#     win32gui.DeleteObject(saveBitMap.GetHandle())
#     saveDC.DeleteDC()
#     mfcDC.DeleteDC()
#     win32gui.ReleaseDC(hWnd, hWndDC)
#     return im2


# def 打印测试数据(数_词表, 数据, 输人_分, 标签):
#     临 = 数据[0]
#     欲打印 = [数_词表[str(临[i])] for i in range(临.size)]
#     打印 = ""
#     for i in range(len(欲打印)):
#         打印 = 打印+欲打印[i]

#     临 = 输人_分.cpu().numpy()[0]
#     欲打印2 = [数_词表[str(临[i])]for i in range(输人_分.size(1))]
#     # 欲打印2=str(欲打印2)
#     # print("输入：", 欲打印2)
#     if 标签 == 打印:
#         return True
#     else:
#         print(打印)
#         return False

#     print("输出：", 打印)

    # for i in range(16):
    #     print(数_词表[str(临[i, 0])])


# def 打印测试数据_A(数_词表, 数据, 输人_分):
#     if 数据.shape[0] != 0:

#         临 = 数据[0]
#         欲打印 = [数_词表[str(临[i])] for i in range(临.size)]
#         打印 = ""
#         for i in range(len(欲打印)):
#             打印 = 打印+欲打印[i]

#         临 = 输人_分.cpu().numpy()[0]
#         欲打印2 = [数_词表[str(临[i])]for i in range(输人_分.size(1))]
#         欲打印2 = str(欲打印2)
#         #print("输入：", 欲打印2)
#         print("输出:", 打印)
