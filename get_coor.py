'''
@File    :   get_coor.py
@Time    :   2021/12/29 22:12:04
@Author  :   goole 
@Version :   1.0
@Discrib :   minitouch要控制的每个按键的获得坐标
'''

import cv2 as cv

maxsize = (2160, 1080)  # 定义图片放缩大小
img = cv.imread(r'C:/Users/DELL/Desktop/MuMu20211226172911.png')
img = cv.resize(img, maxsize, cv.INTER_AREA)



data = {
    '加三技能':'d 0 552 1878 100\nc\nu 0\nc\n',
    '加二技能':'d 0 446 1687 100\nc\nu 0\nc\n',
    '加一技能':'d 0 241 1559 100\nc\nu 0\nc\n',
    '购买':'d 0 651 207 100\nc\nu 0\nc\n',
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
    # "恢复": "d 0 108 1232 100\nc\nu 0\nc\n"
}
# 鼠标事件
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        # 画圈（图像:img，坐标位置:xy，半径:1(就是一个点)，颜色:蓝，厚度：-1(就是实心)
        cv.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv.putText(img, xy, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv.imshow("image", img)
        #写入txt
        
        realx = 1080-y
        realy = x
        # x_str = str(x)
        # y_str = str(y)
        realx = str(realx)
        realy = str(realy)
        
        
        f = open(r"./coordinate.txt", "a+")
        # f.writelines(real + ' ' + y_str + '\n')
        print(': d 0 {} {} 100\nc\nu 0\nc\n'.format( realx, realy))
        # f.writelines('d 0 {} {} 100\nc\nu 0\nc\n'.format( realx, realy))
        f.write('d 0 {} {} 100\nc\nu 0\nc\n'.format( realx, realy))

cv.namedWindow("image")
cv.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv.imshow("image", img)

while (True):
    try:
        cv.waitKey(100)
    except Exception:
        cv.destroyAllWindows()
        break

cv.waitKey(0)
cv.destroyAllWindows()
