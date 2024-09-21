import argparse
import os
import copy
import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from torch.utils.data.dataloader import DataLoader
from datasets import ImageDatasetLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.nn.functional as F

from model import RCAN
# from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calculate_psnr, get_logger, bgr2ycbcr, rgb2ycbcr
from PIL import Image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, default='RCAN')
    
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)

    parser.add_argument('--hr_train_folder', type=str, default='/root/dataset/DIV2K_HR_crop')
    parser.add_argument('--lr_train_folder', type=str,  default='/root/dataset/DIV2K_LR_crop') 
    parser.add_argument('--b0_train_folder', type=str,  default='/root/dataset/DIV2K_LR_crop')       
    parser.add_argument('--hr_eval_folder', type=str, default='/tmp/dataset/Set5_GTmod12')
    parser.add_argument('--lr_eval_folder', type=str,  default='/tmp/dataset/Set5_LR')
    parser.add_argument('--b0_eval_folder', type=str,  default='/tmp/dataset/Set5_LR')

    parser.add_argument('--outputs_dir', type=str,  default='/root/run/rcan/saved_models/DIV2K2')
    parser.add_argument('--preds_dir', type=str,  default='/root/run/rcan/preds_imgs/DIV2K2')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    # parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=78)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=60)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    if not os.path.exists(args.preds_dir):
        os.makedirs(args.preds_dir)

    cudnn.benchmark = True
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = RCAN(args=args)
    # model.to(device)

    torch.manual_seed(args.seed)

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    smodel = RCAN(args=args).to(device_id)
    model = DDP(smodel, device_ids=[dist.get_rank()])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Transformation for the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create Dataset and DataLoader
    hr_train_folder = args.hr_train_folder
    lr_train_folder = args.lr_train_folder
    b0_train_folder = args.b0_train_folder
    train_dataset = ImageDatasetLoss(hr_train_folder, lr_train_folder, b0_train_folder, patch_size=args.patch_size, scale = args.scale, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    hr_eval_folder = args.hr_eval_folder
    lr_eval_folder = args.lr_eval_folder
    b0_eval_folder = args.b0_eval_folder
    eval_dataset = ImageDatasetLoss(hr_eval_folder, lr_eval_folder, b0_eval_folder, patch_size=args.patch_size, scale = args.scale, transform=transform)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    
    logger = get_logger(args.outputs_dir + '/log.log')
    
    logger.info('start training!')
    logger.info('Total Epoch:[{}]\t lr={:.5f}\t batch_size={:.5f}\t num_features={:.5f}\t num_rg={:.5f}\t num_rcab={:.5f}\t  reduction={:.5f}'.format(args.num_epochs , args.lr, args.batch_size, args.num_features, args.num_rg, args.num_rcab, args.reduction))

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch + 1, args.num_epochs))

            for data in train_dataloader:
                # inputs, labels = data

                inputs = data['LR'].to(device_id)
                labels = data['HR'].to(device_id)
                B0 = data['B0'].to(device_id)

                # inputs = data['LR'].to(device_id)
                # labels = data['HR'].to(device_id)

                preds = model(inputs)

                loss = criterion(preds, 2 * labels - B0)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg ))
                t.update(len(inputs))

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch + 1)))

            model.eval()
            # epoch_psnr = AverageMeter()

            psnr_list = []
            psnr_y_list = []
            for data in eval_dataloader:
                inputs = data['LR'].to(device_id)
                labels = data['HR'].to(device_id)
                B0 = data['B0'].to(device_id)

                (imgname, imgext) = os.path.splitext(os.path.basename(data['HR_path'][0]))

                # inputs = data['LR'].to(device_id)
                # labels = data['HR'].to(device_id)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)

                preds = (preds + B0)/2
                labels = np.array(transforms.ToPILImage()(labels[0]))
                preds = np.array(transforms.ToPILImage()(preds[0])) 
                Image.fromarray(preds).save( os.path.join(args.preds_dir, imgname + '_' + str(epoch + 1) + '.png') , 'PNG')

                psnr = calculate_psnr(preds, labels, border=args.scale )
                psnr_y = calculate_psnr( rgb2ycbcr(labels), rgb2ycbcr(preds), border=args.scale )
                psnr_list.append(psnr)
                psnr_y_list.append(psnr_y)

                logger.info('Testing {:20s} - PSNR: {:.2f} dB '
                    'PSNR_Y: {:.2f} dB; '.
                    format(imgname, psnr, psnr_y,)
                    )
            psnr_ave = np.mean(psnr_list)
            psnr_y_ave = np.mean(psnr_y_list)
            if psnr_y > best_psnr:
                best_epoch = epoch
                best_psnr = psnr_y
                best_weights = copy.deepcopy(model.state_dict())           

            # print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
            
            logger.info('Epoch:[{}/{}]\t psnr_ave={:.5f}\t psnr_y_ave={:.5f}'.format(epoch + 1 , args.num_epochs, psnr_ave, psnr_y_ave ))


    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch + 1, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best_'+str(best_epoch + 1) + '_' + str(best_psnr)+'.pth'))
    
    logger.info('finish training!')



if __name__ == '__main__':
    main()


