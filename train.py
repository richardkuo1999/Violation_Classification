import os
import yaml
import math
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

import torch
import torch.optim
import torch.utils.data
from torch.cuda import amp
import torch.backends.cudnn

from utils.loss import Loss
from test import test
from utils.torch_utils import select_device
from utils.datasets import create_dataloader
from models.model import build_model, get_optimizer
from utils.general import colorstr, set_logging, increment_path, write_log,\
                         data_color, AverageMeter
                        #  val_tensorboard, , data_color





logger = logging.getLogger(__name__)



def main(args, hyp, device, writer):
    
    begin_epoch, global_steps, best_fitness, fi = 1, 0, 0.0, 1.0

    # Directories
    logger.info(colorstr('hyperparameter: ') + ', '\
                                .join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, maxEpochs = Path(args.save_dir), args.epochs
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    results_file = save_dir / 'results.txt'

    last = wdir / f'last.pth'
    best = wdir / f'best.pth'


    # Get class and class number
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    DriveArea_class = data_dict['DriveArea_names']
    Lane_class = data_dict['Lane_names']
    logger.info(f"{colorstr('DriveArea_class: ')}{DriveArea_class}")
    logger.info(f"{colorstr('Lane_class: ')}{Lane_class}")

    if args.DoOneHot:
        hyp.update({'nc':[len(Lane_class),len(DriveArea_class)]})
    else:
        hyp.update({'nc':[3,3]})
        


    # Save run settings(hyp, args)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # build up model
    print("begin to build up model...")


    model = build_model(ch=hyp['nc'], num_classes=2, 
                            tokensize=2, split=args.useSplitModel).to(device)

    # loss function 
    criterion = Loss(hyp).to(device)

    # Optimizer
    optimizer = get_optimizer(hyp, model)

    print("finish build model")

    # Data loading
    print("begin to load data")
    normalize = {'mean':[0.485, 0.456, 0.406], 
                    'std':[0.229, 0.224, 0.225]}
    
    train_loader = create_dataloader(args, hyp, data_dict, \
                                        args.batch_size, normalize)
    num_batch = len(train_loader)
    
    valid_loader = create_dataloader(args, hyp, data_dict,\
                                    args.batch_size, normalize, \
                                    is_train=False, shuffle=False)

    print('load data finished')


    lf = lambda x: ((1 + math.cos(x * math.pi / maxEpochs)) / 2) * \
                    (1 - hyp['lrf']) + hyp['lrf']  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # # assign model params
    model.gr = 1.0
    model.nc = 2

    # training
    num_warmup = max(round(hyp['warmup_epochs'] * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    print(colorstr('=> start training...'))
    for epoch in range(begin_epoch, maxEpochs+1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        model.train()
        start = time.time()
        for i, (input, bbox, target, paths, shapes) in enumerate(train_loader, start=1):
            image, laneline, drivable = input
            num_iter = i + num_batch * (epoch - 1)

            if num_iter < num_warmup:
                # warm up
                lf = lambda x: ((1 + math.cos(x * math.pi / maxEpochs)) / 2)* \
                            (1 - hyp['lrf']) + hyp['lrf']  # cosine
                xi = [0, num_warmup]
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  
                # # iou loss ratio (obj_loss = 1.0 or iou)
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, 
                    # all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(num_iter, xi, [hyp['warmup_biase_lr'] \
                                        if j == 2 else 0.0, x['initial_lr'] *\
                                                                lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_iter, xi, 
                                                [hyp['warmup_momentum'], 
                                                    hyp['momentum']])
                        
            data_time.update(time.time() - start)
            image = image.to(device, non_blocking=True)
            laneline = laneline.to(device, non_blocking=True)
            drivable = drivable.to(device, non_blocking=True)
            bbox = bbox.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Forward
            with amp.autocast(enabled=device.type != 'cpu'):
                outputs = model(image, laneline, drivable, bbox)
                loss = criterion(outputs, target)

            # compute gradient and do update step
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # measure accuracy and record loss
            losses.update(loss.item(), image.size(0))
            # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
            #                                  target.detach().cpu().numpy())
            # acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - start)
            if i % 10 == 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                writer.add_scalar('lr', lr, global_steps)
                msg = f'Epoch: [{epoch}][{i}/{len(train_loader)}] '+\
                        f'lr: [{lr}] '
                
                msg +=  '\n                   '+\
                        f'Time {batch_time.sum:.3f}s ({batch_time.avg:.3f}s)  '+\
                        f'peed {image.size(0)/batch_time.val:.1f} samples/s  '+\
                        f'Data {data_time.sum:.3f}s ({data_time.avg:.3f}s)  '+\
                        f'Loss {losses.val:.5f} ({losses.avg:.5f})  '
                logger.info(msg)
                # Write 
                # write_log(results_file, msg)
                # validation result tensorboard
                writer.add_scalar('train_loss', losses.val, global_steps)
                global_steps += 1
            start = time.time()
            
        lr_scheduler.step()


        # evaluate on validation set
        if (epoch >= args.val_start and (epoch % args.val_freq == 0 
                                                    or epoch == maxEpochs)):

            acc, precision, recall, f1Score, total_loss, t= test(
                epoch, args, hyp, valid_loader, model, criterion,save_dir,results_file,device=device)
            # validation result tensorboard
            writer.add_scalar('acc', acc, global_steps)
            writer.add_scalar('val_loss', total_loss, global_steps)


        ckpt = {
            'epoch': epoch,
            'best_fitness': best_fitness,
            'global_steps':global_steps-1,
            'state_dict':  model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        # last
        torch.save(ckpt, last)
        # frequency
        if (epoch % args.val_freq == 0 or epoch == maxEpochs):
            savepath = wdir / f'epoch-{epoch}.pth'
            logger.info(f'{colorstr("=> saving checkpoint")} to {savepath}')
            torch.save(ckpt, savepath)
        # best
        if best_fitness == fi:
            logger.info(f'{colorstr("=> saving checkpoint")} to {savepath}')
            torch.save(ckpt, best)


        del ckpt

    torch.cuda.empty_cache()
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, 
                            default='hyp/hyp.scratch.yolop.yaml', 
                            help='hyperparameter path')
                            # yolop_backbone
    parser.add_argument('--DoOneHot', type=bool, default=False, 
                                            help='do one hot or not')
    parser.add_argument('--useSplitModel', type=bool, default=False, 
                                            help='do one hot or not')
    
    parser.add_argument('--data', type=str, default='data/multi.yaml', 
                                            help='dataset yaml path')
    parser.add_argument('--logDir', type=str, default='runs/train',
                            help='log directory')
    parser.add_argument('--device', default='', 
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16, 
                            help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=0, 
                            help='maximum number of dataloader workers')
    parser.add_argument('--val_start', type=int, default=0, 
                            help='start do validation')
    parser.add_argument('--val_freq', type=int, default=1, 
                            help='How many epochs do one time validation')

    parser.add_argument('--img_size', nargs='+', type=int, default=[256, 256], 
                            help='[train, test] image sizes')

   
    # Cudnn related params
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,  
                                help='Use GPUs to speed up network training')
    parser.add_argument('--cudnn_deterministic', type=bool, default=False, 
                                help='only use deterministic convolution algorithms')
    parser.add_argument('--cudnn_enabled', type=bool, default=True,  
                                help='controls whether cuDNN is enabled')
    return parser.parse_args()

if __name__ == '__main__':
    # Set DDP variables
    args = parse_args()
    set_logging()

    # cudnn related setting
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic

    args.device = select_device(args.device, batch_size=args.batch_size)


    # Hyperparameter
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    args.save_dir = increment_path(Path(args.logDir)) 
    print(args.save_dir)

    # Train
    logger.info(args)
    logger.info(f"{colorstr('tensorboard: ')}Start with 'tensorboard --logdir {args.logDir}'"+\
                                        ", view at http://localhost:6006/")
    writer = SummaryWriter(args.save_dir)  # Tensorboard
    
    main(args, hyp, args.device, writer)