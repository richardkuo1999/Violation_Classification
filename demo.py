import cv2
import yaml
import time
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torchvision.transforms as transforms

from models.model import build_model
from utils.datasets import LoadImages
from utils.torch_utils import select_device, time_synchronized
from utils.plot import plot_one_box
from utils.general import colorstr, increment_path, write_log,\
                        AverageMeter, xywhn2xyxy, OpCounter



logger = logging.getLogger(__name__)

def detect(model, image, laneline, drivable, bbox):
    
    with torch.no_grad():
        
        outputs = model(image, laneline, drivable, bbox)
        
    return outputs

def main(args, device='cpu'):
    
    save_dir = args.save_dir

    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    results_file = save_dir / 'results.txt'
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Get class and class number
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

    #   Det_class = data_dict['Det_names']
    DriveArea_class = data_dict['DriveArea_names']
    Lane_class = data_dict['Lane_names']
    if args.DoOneHot:
        nc = {'nc':[len(Lane_class),len(DriveArea_class)]}
    else:
        nc = {'nc':[3,3]}
    logger.info(f"{colorstr('DriveArea_class: ')}{DriveArea_class}")
    logger.info(f"{colorstr('Lane_class: ')}{Lane_class}")


    # build up model
    print("begin to build up model...")
    model = build_model(ch=nc['nc'], num_classes=2, tokensize=args.tokensize,
                            split=args.useSplitModel).to(device)
    

    # load weights
    checkpoint_file = args.weights
    print("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16
    print('bulid model finished')


    # Data loading
    print("begin to load data")
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    dataset = LoadImages(args, data_dict, transform)
    print('load data finished')

    
       
    # Run inference
    t0 = time.time()
    colors = [(255,255,0), (0,255,255)]
    T_inf = AverageMeter()

    # switch to train mode
    model.eval()
    for batch_i, (input, bbox, paths, shapes) in enumerate(dataset):
        image, laneline, drivable = input

        image = image.to(device, non_blocking=True)
        laneline = laneline.to(device, non_blocking=True)
        drivable = drivable.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)
        image = image.half() if half else image.float()  # uint8 to fp16/32
        laneline = laneline.half() if half else bbox.float()  # uint8 to fp16/32
        drivable = drivable.half() if half else bbox.float()  # uint8 to fp16/32
        bbox = bbox.half() if half else bbox.float()  # uint8 to fp16/32

        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        if laneline.ndimension() == 3:
            laneline = laneline.unsqueeze(0)
        if drivable.ndimension() == 3:
            drivable = drivable.unsqueeze(0)
        if bbox.ndimension() == 1:
            bbox = bbox.unsqueeze(0)

        t = time_synchronized()
        outputs = detect(model, image, laneline, drivable, bbox)
        t_inf = time_synchronized() - t
        T_inf.update(t_inf/image.size(0),image.size(0))

        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        classname = ['violations', 'legitimate']
        cls = outputs.tolist()[0]
        print(f"{paths} result is {classname[cls]}")
        if args.draw:
            imageName = str(Path(paths).parent).replace('images','oriimg') +'/'
            namelist = Path(paths).stem.split("_")
            for name in namelist[:-1]:
                imageName += (name+'_')
            imageName = imageName[:-1] + '.jpg'
            if (save_dir / Path(imageName).name).exists():
                imageName = save_dir / Path(imageName).name
            oriimg = cv2.imread(str(imageName))
            h, w = oriimg.shape[:-1]
            xyxy = xywhn2xyxy(bbox, w, h)[0]
            plot_one_box(xyxy, oriimg , label=classname[cls], color=colors[int(cls)], line_thickness=4)
            cv2.imwrite(str(save_dir / Path(imageName).name),oriimg)

    # calculate macs, params, flops, parameter count
    img = torch.rand(3, 256, 256).to(device)
    if args.DoOneHot:
        laneline = torch.rand(25, 256, 256).to(device)
        drivable = torch.rand(2, 256, 256).to(device)
    else:
        laneline = torch.rand(3, 256, 256).to(device)
        drivable = torch.rand(3, 256, 256).to(device)
    bbox = torch.rand(1,4).to(device)

    img = img.half() if half else img.float()  # uint8 to fp16/32
    laneline = laneline.half() if half else laneline.float()  # uint8 to fp16/32
    drivable = drivable.half() if half else drivable.float()  # uint8 to fp16/32
    bbox = bbox.half() if half else bbox.float()  # uint8 to fp16/32
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    if laneline.ndimension() == 3:
        laneline = laneline.unsqueeze(0)
    if drivable.ndimension() == 3:
        drivable = drivable.unsqueeze(0)
    if bbox.ndimension() == 1:
        bbox = bbox.unsqueeze(0)

    OpCounter(img, laneline, drivable, bbox, model, 
              results_file, split=args.useSplitModel)
    
    msg = f'{str(args.weights)}\n'+\
          f'Results saved to {str(args.save_dir)}\n'+\
          f'Done. ({(time.time() - t0)} s)\n'+\
          f'inf : ({T_inf.avg} s/frame)'+\
          f'fps : ({(1/(T_inf.avg))} frame/s)'
    write_log(results_file, msg)



def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network')
    parser.add_argument('--hyp', type=str, default='hyp/hyp.scratch.yolop.yaml', 
                            help='hyperparameter path')
    parser.add_argument('--DoOneHot', type=bool, default=True, 
                                            help='do one hot or not')
    parser.add_argument('--useSplitModel', type=bool, default=False, 
                                            help='do one hot or not')
    
    parser.add_argument('--data', type=str, default='data/multi.yaml', 
                                            help='dataset yaml path')
    parser.add_argument('--tokensize', type=int, default=32, 
                                            help='size of the tokens')
    parser.add_argument('--logDir', type=str, default='runs/demo',
                            help='log directory')
    parser.add_argument('--source', type=str, default='./inference/val', 
                                                    help='source')  
    parser.add_argument('--img_size', nargs='+', type=int, default=[256, 256], 
                            help='[train, test] image sizes')
    parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default='weights/MO.pth', 
                                                    help='model.pth path(s)')
    parser.add_argument('--draw', type=bool, default=False, 
                                                    help='draw result')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    args.device = select_device(args.device)

    args.save_dir = increment_path(Path(args.logDir))  # increment run
    main(args, args.device)

    print("detect finish")