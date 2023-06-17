import cv2
import yaml
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from prefetch_generator import BackgroundGenerator

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from utils.general import one_hot_it_v11_dice
from utils.augmentations import letterbox

def create_dataloader(args, hyp, data_dict, batch_size, normalize, is_train=True, shuffle=True):
    normalize = transforms.Normalize(
            normalize['mean'], normalize['std']
        )
    
    dataset = MyDataset(
            args=args,
            hyp=hyp,
            data_dict=data_dict,
            dataSet=data_dict['train'] if is_train else data_dict['val'],
            is_train=is_train,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

    dataloader = DataLoaderX(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=args.workers,
                            pin_memory=True,
                            collate_fn=MyDataset.collate_fn
                            )

    return dataloader

class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class MyDataset(Dataset):
    def __init__(self, args, hyp, data_dict, dataSet, is_train, transform=None):
        """
        initial all the characteristic

        Inputs:
        -args: configurations
        -transform: ToTensor and Normalize

        Returns:
        None
        """
        self.hyp = hyp
        self.DoOneHot = args.DoOneHot
        self.is_train = is_train
        self.device = args.device
        self.transform = transform
        self.inputsize = args.img_size
        self.Tensor = transforms.ToTensor()

        # Data Root
        self.img_root = Path(dataSet[0])
        self.DriveArea_root = Path(dataSet[1])
        self.laneline_root = Path(dataSet[2])
        self.object_root = Path(dataSet[3])
        # Data class
        self.label_Lane_info = data_dict['Lane_names']
        self.label_drivable_info = data_dict['DriveArea_names']

        self.img_list = self.img_root.iterdir()

        self.db = self.__get_db()

    def __len__(self):
        return len(self.db)
         
    def __get_db(self):
        """get database from the annotation file

        Returns:
            gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'label':, 'mask':,'lane':}
            image: image path
            label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
            mask: path of the driver area segmentation label path
            lane: path of the lane segmentation label path
        """
        gt_db = []
        for img_path in tqdm(list(self.img_list)):
            image_path = str(img_path)
            DriveArea_path = image_path.replace(str(self.img_root), 
                                str(self.DriveArea_root)).replace(".jpg", ".png")
            lane_path = image_path.replace(str(self.img_root), 
                                str(self.laneline_root)).replace(".jpg", ".png")
            label_path = image_path.replace(str(self.img_root), 
                                str(self.object_root)).replace(".jpg", ".json")




            with open(label_path, 'r') as f:
                label = json.load(f)
            obj = label['box2dxywh']
            xywh = torch.tensor([float(obj['x']),float(obj['y']),
                    float(obj['w']),float(obj['h'])])
            gt = 1 if label['category'] == 'legitimate' else 0
            label = torch.zeros(2)
            label[gt] = 1
           

            rec = [{
                'image': image_path,
                'DriveArea': DriveArea_path,
                'lane': lane_path,
                'xywh': xywh,
                'label': label
            }]

            gt_db += rec
        return gt_db


    def __getitem__(self, idx):
        hyp = self.hyp
        data = self.db[idx]
        resized_shape = max(self.inputsize) if isinstance(self.inputsize, list) \
                                            else self.inputsize

        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]  # orig hw

        drivable_label = cv2.imread(data["DriveArea"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        drivable_label = cv2.cvtColor(drivable_label, cv2.COLOR_BGR2RGB)
        lane_label = cv2.imread(data["lane"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        lane_label = cv2.cvtColor(lane_label, cv2.COLOR_BGR2RGB)

        #resize
        (img, drivable_label, lane_label), ratio, pad = letterbox((img, drivable_label, lane_label),\
                                         resized_shape, auto=False, scaleup=self.is_train)
        h, w = img.shape[:2]

        if self.DoOneHot:
            drivable_label = one_hot_it_v11_dice(drivable_label, self.label_drivable_info)
            lane_label = one_hot_it_v11_dice(lane_label, self.label_Lane_info)
            # self.segement_debug(img, drivable_label, lane_label, idx, data)
            drivable_label = torch.tensor(drivable_label)
            lane_label = torch.tensor(lane_label)
        else:
            drivable_label = self.Tensor(drivable_label)
            lane_label = self.Tensor(lane_label)
        img = self.transform(img)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        return img, lane_label, drivable_label, data["xywh"], data["label"], data["image"], shapes 
    
    def segement_debug(self, img, drivable_label, lane_label, idx, data):
        from PIL import Image
        # aaa = img.copy()
        aaa = np.zeros(img.shape,dtype=img.dtype)
        drivable_label_bool = drivable_label.copy().astype(dtype=bool)
        for i in range(1,len(drivable_label_bool)):
            aaa[drivable_label_bool[i,:,:]] = self.label_drivable_info[list(self.label_drivable_info)[i]][:3]

        lane_label_bool = lane_label.copy().astype(dtype=bool)
        for i in range(1,len(lane_label_bool)):
            aaa[lane_label_bool[i,:,:]] = self.label_Lane_info[list(self.label_Lane_info)[i]][:3]
        aaa = cv2.cvtColor(aaa, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'runs/{idx}.png',aaa)
        cv2.imwrite(f'runs/{idx}_.png',img)
        print(data["image"])
        print(data["DriveArea"])
        print(data["lane"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # TODO collate_fn
    @staticmethod
    def collate_fn(batch):
        image, lane, drivable, bbox, target, paths, shapes= zip(*batch)
        return [torch.stack(image, 0), torch.stack(lane, 0), torch.stack(drivable, 0)], \
                            torch.stack(bbox, 0), torch.stack(target, 0), paths, shapes



class LoadImages:  # for inference
    def __init__(self, args, data_dict, transform):
        self.inputsize = args.img_size
        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.device = args.device
        self.DoOneHot = args.DoOneHot
        # Data Root
        self.img_root = Path(args.source) / 'images'
        self.DriveArea_root = Path(args.source) / 'DriveArea'
        self.laneline_root = Path(args.source) / 'laneline'
        self.object_root = Path(args.source) / 'object'
        self.label_Lane_info = data_dict['Lane_names']
        self.label_drivable_info = data_dict['DriveArea_names']
        
        self.img_list = self.img_root.iterdir()
        self.db = self.__get_db()

    def __len__(self):
        return len(self.db)  # number of files

    def __get_db(self):
        """get database from the annotation file

        Returns:
            gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'label':, 'mask':,'lane':}
            image: image path
            label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
            mask: path of the driver area segmentation label path
            lane: path of the lane segmentation label path
        """
        gt_db = []
        for img_path in tqdm(list(self.img_list)):
            image_path = str(img_path)
            DriveArea_path = image_path.replace(str(self.img_root), 
                                str(self.DriveArea_root)).replace(".jpg", ".png")
            lane_path = image_path.replace(str(self.img_root), 
                                str(self.laneline_root)).replace(".jpg", ".png")
            label_path = image_path.replace(str(self.img_root), 
                                str(self.object_root)).replace(".jpg", ".json")


            with open(label_path, 'r') as f:
                label = json.load(f)
            obj = label['box2dxywh']
            xywh = torch.tensor([float(obj['x']),float(obj['y']),
                    float(obj['w']),float(obj['h'])])

            rec = [{
                'image': image_path,
                'DriveArea': DriveArea_path,
                'lane': lane_path,
                'xywh': xywh,
            }]

            gt_db += rec
        return gt_db


    def __getitem__(self, idx):
        data = self.db[idx]
        resized_shape = max(self.inputsize) if isinstance(self.inputsize, list) \
                                            else self.inputsize

        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]  # orig hw

        drivable_label = cv2.imread(data["DriveArea"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        drivable_label = cv2.cvtColor(drivable_label, cv2.COLOR_BGR2RGB)
        lane_label = cv2.imread(data["lane"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        lane_label = cv2.cvtColor(lane_label, cv2.COLOR_BGR2RGB)

        #resize
        (img, drivable_label, lane_label), ratio, pad = letterbox((img, drivable_label, lane_label),\
                                         resized_shape, auto=False)
        h, w = img.shape[:2]

        if self.DoOneHot:
            drivable_label = one_hot_it_v11_dice(drivable_label, self.label_drivable_info)
            lane_label = one_hot_it_v11_dice(lane_label, self.label_Lane_info)
            # self.segement_debug(img, drivable_label, lane_label, idx, data)
            drivable_label = torch.tensor(drivable_label)
            lane_label = torch.tensor(lane_label)
        else:
            drivable_label = self.Tensor(drivable_label)
            lane_label = self.Tensor(lane_label)

        img = self.transform(img)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        bbox = data["xywh"]

        return [img, lane_label, drivable_label], bbox, data["image"], shapes

    def segement_debug(self, img, drivable_label, lane_label, idx, data):
        from PIL import Image
        # aaa = img.copy()
        aaa = np.zeros(img.shape,dtype=img.dtype)
        drivable_label_bool = drivable_label.copy().astype(dtype=bool)
        for i in range(1,len(drivable_label_bool)):
            aaa[drivable_label_bool[i,:,:]] = self.label_drivable_info[list(self.label_drivable_info)[i]][:3]

        lane_label_bool = lane_label.copy().astype(dtype=bool)
        for i in range(1,len(lane_label_bool)):
            aaa[lane_label_bool[i,:,:]] = self.label_Lane_info[list(self.label_Lane_info)[i]][:3]
        aaa = cv2.cvtColor(aaa, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'runs/{idx}.png',aaa)
        cv2.imwrite(f'runs/{idx}_.png',img)
        print(data["image"])
        print(data["DriveArea"])
        print(data["lane"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # TODO collate_fn
    @staticmethod
    def collate_fn(batch):
        image, lane, drivable, bbox, paths, shapes= zip(*batch)
        return [torch.stack(image, 0), torch.stack(lane, 0), torch.stack(drivable, 0)], \
                            torch.stack(bbox, 0), paths, shapes






if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description='Test Multitask network')
        parser.add_argument('--data', type=str, default='F:/ITRI/classify/data/multi.yaml', 
                                                help='dataset yaml path')
        parser.add_argument('--source', type=str, default='F:/ITRI/classify/inference/val', 
                                                        help='source')  
        parser.add_argument('--img_size', nargs='+', type=int, default=[224, 224], 
                                help='[train, test] image sizes')
        return parser.parse_args()


    args = parse_args()

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    
     # Get class and class number
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

    dataset = LoadImages(args, data_dict, transform)
    for batch_i, (image, bbox, paths, shapes) in enumerate(dataset):
        print(batch_i)
        pass