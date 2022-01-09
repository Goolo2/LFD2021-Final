import torch
import torchvision
import numpy as np
import os
import json
from PIL import Image
from resnet_utils import myResnet

records = '../dataset/unused'
if not os.path.exists(records):
    os.makedirs(records)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
resnet101 = torchvision.models.resnet101(pretrained=True).eval()
resnet101 = myResnet(resnet101).cuda(device).requires_grad_(False)
comb_idx = "./json/comb_idx.json"

with open(comb_idx, encoding='utf8') as f:
    comb_dict = json.load(f)

for root, dirs, files in os.walk(records):
    if len(dirs) > 0:
        break
for folder in dirs:
    # jsondir = records+'/' + folder + '/operations.json'
    jsondir = records+'/' + folder + '/newoperations.json'
    npdata_dir = records+'/' + folder + '/processed_data.npz'
    if os.path.isfile(npdata_dir):
        continue

    img_tensor = torch.Tensor(0)

    # print(img_tensor.shape[0])
    ope_tensor = torch.Tensor(0)

    ope_seq = np.ones((1, 1))
    end_seq = np.ones((1, 1))
    count = 0
    print('processing folder ... {}'.format(folder))
    data_col = []
    with open(jsondir, encoding='ansi') as f:
        while True:
            df = f.readline()
            df = df.replace('\'', '\"')

            if df == "":
                break
            df = json.loads(df)
            data_col.append(df)

    with open(jsondir, encoding='ansi') as f:
        move_ope = '无移动'
        for i in range(len(data_col)):
            df = data_col[i]

            if img_tensor.shape[0] == 0:
                img = Image.open(records+'/' + folder + '/{}.jpg'.format(df["img_idx"]))
                img2 = np.array(img)

                # img2 = torch.tensor(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _, out = resnet101(img2)
                img_tensor = out.reshape(1, 6*6*2048)
                tmp_move = df["move_ope"]
                if tmp_move != '无移动':
                    move_ope = tmp_move

                ope_seq[0, 0] = comb_dict[move_ope + "_" + df["act_ope"]]
                end_seq[0, 0] = df["end"]
            else:
                img = Image.open(records+'/' + folder + '/{}.jpg'.format(df["img_idx"]))
                img2 = np.array(img)

                # img2 = torch.tensor(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _, out = resnet101(img2)

                img_tensor = torch.cat((img_tensor, out.reshape(1, 6*6*2048)), 0)
                tmp_move = df["move_ope"]
                if tmp_move != '无移动':
                    move_ope = tmp_move
                ope_seq = np.append(ope_seq, comb_dict[move_ope + "_" + df["act_ope"]])
                end_seq = np.append(end_seq, df["end"])

        img_tensornp = img_tensor.cpu().numpy()
        ope_seq = ope_seq.astype(np.int64)
        np.savez(npdata_dir, img_tensornp=img_tensornp, ope_seq=ope_seq, end_seq=end_seq)
