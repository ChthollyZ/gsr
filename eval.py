import os
import argparse
from PIL import Image
from utils import calc_psnr
import numpy as np
# import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms
from model import RCAN
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import ImageDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, default='RCAN')
    # parser.add_argument('--images_dir', type=str, required=True)
    # parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--hr_eval_folder', type=str, default='/tmp/dataset/0801/HR_test')
    parser.add_argument('--lr_eval_folder', type=str,  default='/tmp/dataset/0801/LR_test')
    args = parser.parse_args()


    hr_eval_folder = '/tmp/dataset/0801/HR_test'
    lr_eval_folder ='/tmp/dataset/0801/LR_test'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    eval_dataset = ImageDataset(hr_eval_folder, lr_eval_folder, transform=transform)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    model = RCAN(args=args)

    path = '/root/run/rcan/saved_models/0801_HR1/x4/epoch_'

    lr0_list = []
    lr1_list = []

    for i in range(1, 6):
        print(i)
        j = str(i)
        # 加载state_dict，假设state_dict是从DataParallel模型保存的
        state_dict_path = path + j + '.pth'
        state_dict = torch.load(state_dict_path)

        # 由于state_dict是从DataParallel模型保存的，我们需要处理module关键字
        # 可以通过以下方式删除module前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:]  # 去掉'module.'前缀
            new_state_dict[name] = v

        # 加载处理过的state_dict到模型
        model.load_state_dict(new_state_dict, strict=True)

        # 将模型移动到GPU上（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        model.eval()

        for data in eval_dataloader:
            inputs = data['LR'].to(device)
            labels = data['HR'].to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
            p = calc_psnr(preds, labels).cpu().numpy()
            # print(data['LR_path'], p)
            if data['LR_path'][0][-5:] == '0.png':
                print('lr0', ' PSNR: ', p)
                lr0_list.append(p)
            else:
                print('lr1', ' PSNR: ', p)
                lr1_list.append(p)
            # print(data['LR_path'][0] , ' PSNR: ', p)

    print('LR0', np.array(lr0_list))
    print('LR1', np.array(lr1_list))