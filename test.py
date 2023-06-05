import yaml
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

import torch


from utils.loss import Loss
from models.model import build_model
from utils.datasets import create_dataloader
from utils.torch_utils import select_device, time_synchronized
from utils.metrics import ClassifyMetric
from utils.general import colorstr, increment_path, write_log,\
                        data_color, AverageMeter



logger = logging.getLogger(__name__)

def test(epoch, args, hyp, val_loader, model, criterion, output_dir,
              results_file, logger=None, device='cpu'):
    # save_dir = output_dir / 'visualization'
    # save_dir.mkdir(parents=True, exist_ok=True)
    # save_dir = str(save_dir)

    batch_size = args.batch_size
    seen =  0 

    #detector confusion matrix
    metric = ClassifyMetric(2,[0, 1]) 

    losses = AverageMeter()
    T_inf = AverageMeter()

    # switch to train mode
    model.eval()
    for batch_i, (input, bbox, target, paths, shapes) in enumerate(tqdm(val_loader)):
        image, laneline, drivable = input
        image = image.to(device, non_blocking=True)
        laneline = laneline.to(device, non_blocking=True)
        drivable = drivable.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w, pad_h = int(pad_w), int(pad_h)
            t = time_synchronized()
            outputs = model(image, laneline, drivable, bbox)
            t_inf = time_synchronized() - t


            if batch_i > 0:
                T_inf.update(t_inf/image.size(0),image.size(0))

            loss = criterion(outputs, target)
            losses.update(loss.item(), image.size(0))
        
        outputs = torch.argmax(outputs, dim=1)
        target = torch.argmax(target, dim=1)
        metric.addBatch(outputs.cpu(), target.cpu())


    # Print speeds
    t = tuple((t_inf, batch_size))  # tuple
    
    print('Speed: %.1f ms inference/total per image at batch-size %g' % t)

    model.float()  # for training
    t = T_inf.avg

    # Print results
    acc = metric.accuracy()
    precision = metric.precision()
    recall = metric.recall()
    f1Score = metric.f1_score()
    msg = f'Epoch: [{epoch}]    Loss({losses.avg:.3f})\nDetect:\n'+\
          f'    acc:    {acc:.3f}\n'+\
          f'    f1score:    violations:{f1Score[0]:.3f}   legitimate:{f1Score[1]:.3f}\n'+\
          f'    precision:    violations:{precision[0]:.3f}   legitimate:{precision[1]:.3f}\n'+\
          f'    recall:    violations:{recall[0]:.3f}   legitimate:{recall[1]:.3f}\n'+\
          f'    Time: inference({t:.4f}s/frame)'
    if(logger):
        logger.info(msg)
    write_log(results_file, msg)

    return acc, precision, recall, f1Score, losses.avg, t



def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network')
    parser.add_argument('--hyp', type=str, default='hyp/hyp.scratch.yolop.yaml', 
                            help='hyperparameter path')
    parser.add_argument('--data', type=str, default='data/multi.yaml', 
                                            help='dataset yaml path')
    parser.add_argument('--logDir', type=str, default='runs/test',
                            help='log directory')
    parser.add_argument('--img_size', nargs='+', type=int, default=[224, 224], 
                            help='[train, test] image sizes')
    parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default='weights/last.pth', 
                                                        help='model.pth path(s)')
    parser.add_argument('--batch_size', type=int, default=15, 
                            help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=0, 
                            help='maximum number of dataloader workers')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    args.device = select_device(args.device, batch_size=args.batch_size)


    # Hyperparameter
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Directories
    args.save_dir = Path(increment_path(Path(args.logDir)))  # increment run
    results_file = args.save_dir / 'results.txt'
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Get class and class number
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

    DriveArea_class = data_dict['DriveArea_names']
    Lane_class = data_dict['Lane_names']
    hyp.update({'nc':[len(Lane_class),len(DriveArea_class)]})
    logger.info(f"{colorstr('DriveArea_class: ')}{DriveArea_class}")
    logger.info(f"{colorstr('Lane_class: ')}{Lane_class}")


    # build up model
    print("begin to build up model...")
    model = build_model(ch=hyp['nc'], num_classes=2, 
                            tokensize=32).to(args.device)
    
    # loss function 
    criterion = Loss(hyp).to(args.device)

    # load weights
    model_dict = model.state_dict()
    checkpoint_file = args.weights
    print("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location= args.device)
    checkpoint_dict = checkpoint['state_dict']
    # checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
    model_dict.update(checkpoint_dict)

    model.load_state_dict(model_dict)
    print("=> loaded checkpoint '{}' ".format(checkpoint_file))

    model = model.to(args.device)
    model.gr = 1.0
    model.nc = 2
    print('bulid model finished')

    epoch = checkpoint['epoch'] #special for test

    # Data loading
    print("begin to load data")
    normalize = {'mean':[0.485, 0.456, 0.406], 
                 'std':[0.229, 0.224, 0.225]}
    
    valid_loader = create_dataloader(args, hyp, data_dict,\
                                    args.batch_size, normalize, \
                                    is_train=False, shuffle=False)
    print('load data finished')

    # Save run settings
    with open(args.save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(args.save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    test(epoch, args, hyp, valid_loader, model, criterion, args.save_dir,
              results_file, device=args.device)

    print("test finish")