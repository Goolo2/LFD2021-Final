import os
import torchvision
from Batch import create_masks
from utils import combine_states

from get_trainingdata import *
from utils import *

from resnet_utils import myResnet
from 运行辅助 import *
from pynput.keyboard import Controller, Key, Listener
from pynput import keyboard
import time
import threading
from Model_strategy import Agent
# _DEVICE_ID = '68UDU17B14011947'
# _DEVICE_ID = 'd1cc0a52' #小米
# _DEVICE_ID = 'emulator-5554' #雷电
_DEVICE_ID = '127.0.0.1:7555'  # mumu


# window="RNE-AL00"
# window="PCLM10" #雷电
window = "R11"  # mumu

datadir = '../dataset/unused'
if not os.path.exists(datadir):
    os.makedirs(datadir)
lock = threading.Lock()
start = time.time()
end = time.time()
fun_start = 0
time_interval = 0
index = 0
dict = {'interval_times': 0, 'max_interval': 0., 'interval_location': []}
count = 0
count_dict = {'first_time': 0., 'first_p_to_second_r': 0.}
keyBoard_dict = {'Key.enter': '\n',
                 'Key.space': ' ',
                 "Key.tab": '\t'}

pressW = False
pressS = False
pressA = False
pressD = False
pressQ = False
attack_mode = False
handon_mode = False
attack_rel = True
AI_flag = True
ope_list = []
auto = 0

N = 15000  # 运行N次后学习
parallel = 100
episode = 3
lr = 0.0003
agent = Agent(动作数=7, 并行条目数=parallel,
              学习率=lr, 轮数=episode,
              输入维度=6)


def get_key_name(key):
    if isinstance(key, keyboard.KeyCode):
        return key.char

    else:
        return str(key)
# 监听按压


def on_press(key):
    global fun_start, time_interval, index, dict, count, count_dict, pressW, pressS, pressA, pressD, handon_mode, ope_list, AI_flag, attack_rel, pressQ, attack_mode

    key_name = get_key_name(key)
    operaction = ''

    if key_name == 'w':
        pressW = True
    elif key_name == 'a':
        pressA = True
    elif key_name == 's':
        pressS = True
    elif key_name == 'd':
        pressD = True
    elif key_name == 'q':
        pressQ = True
    elif key_name == 'i':
        AI_flag = bool(1 - AI_flag)

    elif key_name == 'Key.space':
        operaction = '召唤师技能'
    elif key_name == 'Key.end':
        operaction = '补刀'
    elif key_name == 'Key.page_down':
        operaction = '推塔'
    elif key_name == 'j':
        operaction = '一技能'
    elif key_name == 'k':
        operaction = '二技能'
    elif key_name == 'l':
        operaction = '三技能'
    elif key_name == 'f':
        operaction = '回城'
    elif key_name == 'g':
        operaction = '恢复'
    elif key_name == 'h':
        operaction = '召唤师技能'
    elif key_name == 'Key.left':
        operaction = '一技能'
    elif key_name == 'Key.down':
        operaction = '二技能'
    elif key_name == 'Key.right':
        operaction = '三技能'
    elif key_name == 'Key.up':
        attack_mode = True

    lock.acquire()
    if operaction != '':
        ope_list.append(operaction)
    lock.release()
    #print("正在按压:", key_name)

# 监听释放


def on_release(key):
    global start, fun_start, time_interval, index, count, count_dict, pressW, pressS, pressA, pressD, attack_rel, pressQ, attack_mode

    key_name = get_key_name(key)
    if key_name == 'w':
        pressW = False
    elif key_name == 'a':
        pressA = False
    elif key_name == 's':
        pressS = False
    elif key_name == 'd':
        pressD = False
    elif key_name == 'q':
        pressQ = False

    elif key_name == 'Key.up':

        attack_mode = False
    print("已经释放:", key_name)
    if key == Key.esc:
        # 停止监听
        return False

# 开始监听


def start_listen():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def get_direction():
    # W键按下 = False
    # S键按下 = False
    # A键按下 = False
    # D键按下 = False
    if pressQ == True:
        return ('移动停')
    elif pressW == True and pressS == False and pressA == False and pressD == False:
        return ('上移')
    elif pressW == False and pressS == True and pressA == False and pressD == False:
        return ('下移')
    elif pressW == False and pressS == False and pressA == True and pressD == False:
        return ('左移')
    elif pressW == False and pressS == False and pressA == False and pressD == True:
        return ('右移')
    elif pressW == True and pressS == False and pressA == True and pressD == False:
        return ('左上移')
    elif pressW == True and pressS == False and pressA == False and pressD == True:
        return ('右上移')
    elif pressW == False and pressS == True and pressA == True and pressD == False:
        return ('左下移')
    elif pressW == False and pressS == True and pressA == False and pressD == True:
        return ('右下移')
    else:
        return ('')


add_third_skill = 'd 0 559 1767 100\nc\nu 0\nc\n'
add_sec_skill = 'd 0 443 1562 100\nc\nu 0\nc\n'
add_fst_skill = 'd 0 246 1448 100\nc\nu 0\nc\n'
buy = 'd 0 636 190 100\nc\nu 0\nc\n'
# 词数词典路径="./json/词_数表.json"
# 数_词表路径="./json/数_词表.json"

ope_dict = {"img_idx": "0", "move_ope": "无移动", "act_ope": "无动作"}
th = threading.Thread(target=start_listen,)
th.start()  # 启动线程

# if os.path.isfile(词数词典路径) and os.path.isfile(数_词表路径):
#     词_数表, idx_comb = 读出引索(词数词典路径, 数_词表路径)


comb_idx_dir = "./json/comb_idx.json"
idx_comb_dir = "./json/idx_comb.json"
ope_com_dir = "./json/ope_command.json"

comb_idx = read_json(comb_idx_dir)
idx_comb = read_json(idx_comb_dir)
ope_command_dict = read_json(ope_com_dir)

# with open(ope_com_dir, encoding='utf8') as f:
#     ope_command_dict = json.load(f)


direct_sheet = ['上移', '下移', '左移', '右移', '左上移', '左下移', '右上移', '右下移']


simulator = MyMNTDevice(_DEVICE_ID)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
mod = torchvision.models.resnet101(pretrained=True).eval().cuda(device).requires_grad_(False)
resnet101 = myResnet(mod)


while True:
    if AI_flag:
        img_dir = datadir+'/{}/'.format(str(int(time.time())))
        os.mkdir(img_dir)

        record_file = open(img_dir+'_操作数据.json', 'w+')

        img_tensor = torch.Tensor(0)
        ope_tensor = torch.Tensor(0)

        ope_seq = np.ones((1, ))
        ope_seq[0] = 128
        count = 0
        time_start = time.time()
        old_command = '移动停'

        for i in range(1000000):
            if AI_flag == False:
                break
            try:
                imgA = capture_img(window)
            except:
                AI_flag = False
                print('取图失败')
                break

            start_t = time.time()

            if img_tensor.shape[0] == 0:

                img = np.array(imgA)

                img = torch.tensor(img).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _, out = resnet101(img)
                img_tensor = out.reshape(1, 6*6*2048)

            elif img_tensor.shape[0] < 300:

                img = np.array(imgA)

                img = torch.tensor(img).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _, out = resnet101(img)
                img_tensor = torch.cat((img_tensor, out.reshape(1, 6*6*2048)), 0)
                ope_seq = np.append(ope_seq, action)

            else:

                img = np.array(imgA)

                img = torch.tensor(img).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _, out = resnet101(img)
                img_tensor = img_tensor[1:300, :]
                ope_seq = ope_seq[1:300]
                ope_seq = np.append(ope_seq, action)
                img_tensor = torch.cat((img_tensor, out.reshape(1, 6*6*2048)), 0)

            ope_tensor = torch.tensor(ope_seq.astype(np.int64)).cuda(device)
            src_mask, trg_mask = create_masks(ope_tensor.unsqueeze(0), ope_tensor.unsqueeze(0), device)

            state = combine_states(img_tensor.cpu().numpy(), ope_seq, trg_mask)

            action, action_prob, critic = agent.select_action(state, device, 1, False)

            LI = ope_tensor.contiguous().view(-1)
            # LA=输出_实际_A.view(-1, 输出_实际_A.size(-1))
            if count % 50 == 0 and count != 0:

                simulator.send_command(buy)
                simulator.send_command(add_third_skill)
                simulator.send_command(add_sec_skill)
                simulator.send_command(add_fst_skill)
                simulator.send_command(ope_command_dict['移动停'])
                print(old_command, '周期')
                time.sleep(0.02)
                simulator.send_command(ope_command_dict[old_command])

            if count % 1 == 0:
                time_end = time.time()

                command = idx_comb[str(action)]
                command_set = command.split('_')

                #操作词典 = {"img_idx": "0", "move_ope": "无移动", "act_ope": "无动作"}
                ope_dict['img_idx'] = str(i)
                direction = get_direction()

                if direction != '' or len(ope_list) != 0 or attack_mode == True:
                    if direction == '':
                        ope_dict['move_ope'] = command_set[0]
                    else:
                        ope_dict['move_ope'] = direction

                    if len(ope_list) != 0:
                        ope_dict['act_ope'] = ope_list[0]
                        lock.acquire()
                        del ope_list[0]
                        lock.release()
                    elif attack_mode == True:
                        ope_dict['act_ope'] = '攻击'

                    else:
                        ope_dict['act_ope'] = '无动作'

                    savedir = img_dir + '{}.jpg'.format(str(i))
                    imgA.save(savedir)
                    if auto == 0:
                        ope_dict['结束'] = 1
                    else:
                        ope_dict['结束'] = 0
                    auto = 1
                    json.dump(ope_dict, record_file, ensure_ascii=False)
                    record_file.write('\n')

                    new_command = ope_dict['move_ope']
                    if new_command != old_command and new_command != '无移动':
                        old_command = new_command
                        # print(旧指令,操作查询词典[旧指令])
                        try:
                            print('手动模式', old_command)

                            simulator.send_command(ope_command_dict[old_command])

                        except:
                            AI_flag = False
                            print('发送失败')
                            break

                        time.sleep(0.01)

                    if ope_dict['act_ope'] != '无动作' and ope_dict['act_ope'] != '发起集合' and ope_dict['act_ope'] != '发起进攻' and ope_dict['act_ope'] != '发起撤退':
                        print('手动', command_set[1])
                        try:
                            simulator.send_command(ope_command_dict[ope_dict['act_ope']])
                        except:
                            AI_flag = False
                            print('发送失败')
                            break
                else:
                    ope_list = []
                    ope_dict['move_ope'] = command_set[0]
                    ope_dict['act_ope'] = command_set[1]

                    new_command = command_set[0]
                    if new_command != old_command and new_command != '无移动':
                        old_command = new_command
                        # print(旧指令,操作查询词典[旧指令])
                        try:
                            print(old_command)

                            simulator.send_command(ope_command_dict[old_command])

                        except:
                            AI_flag = False
                            print('发送失败')
                            break

                        time.sleep(0.01)
                    savedir = img_dir + '{}.jpg'.format(str(i))
                    imgA.save(savedir)
                    auto = 0
                    ope_dict['结束'] = 0
                    json.dump(ope_dict, record_file, ensure_ascii=False)
                    record_file.write('\n')

                    new_command = ope_dict['move_ope']
                    if command_set[1] != '无动作' and command_set[1] != '发起集合' and command_set[1] != '发起进攻' and command_set[1] != '发起撤退':
                        print(command_set[1])
                        try:
                            simulator.send_command(ope_command_dict[command_set[1]])
                        except:
                            AI_flag = False
                            print('发送失败')
                            break
                time_t1 = 0.22-(time.time()-start_t)
                if time_t1 > 0:
                    time.sleep(time_t1)

                # print(用时1)
                time_t2 = time_end - time_start
                #print("用时{} 第{}张 延时{}".format(用时, i,用时1),'A键按下', A键按下, 'W键按下', W键按下, 'S键按下', S键按下, 'D键按下', D键按下, '旧指令', 旧指令, 'AI打开', AI打开, '操作列', 操作列)

                count = count+1
                if i % 3000 == 0:
                    time.sleep(1)

    record_file.close()
    time.sleep(1)
    print('AI打开', AI_flag)
