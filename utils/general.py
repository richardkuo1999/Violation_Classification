import cv2
import re
import glob
import time
import math
import torch
import logging
import numpy as np
from pathlib import Path

import pandas as pd
from thop import profile
from torchstat import stat
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        
def set_logging():
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO)

def write_log(results_file, msg):
    print(msg)
    with open(results_file, 'a') as f:
        f.write(msg+'\n')  

def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *opt, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in opt) + f'{string}' + colors['end']

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    path = path / time_str
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return path
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return path / sep / n  # update path

def one_hot_it_v11_dice(label, label_info):
    # return semantic_map -> [class_num, H, W]
    height, width, _ = label.shape
    class_num = len(label_info)

    semantic_map = np.zeros((class_num, height, width), dtype=np.float32)

    for index, info in enumerate(label_info):
        if index == 0: continue
        color = label_info[info][:3]
        equality = np.all(label == color, axis=-1)
        semantic_map[index, ...] = equality.astype(np.float32)

    return semantic_map

def data_color(label_info):
    color = []
    for index, info in enumerate(label_info):
        color.append(np.array(label_info[info][:3]))
    return color

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def value_to_float(x):
    if type(x) == float or type(x) == int or not x:
        return x
    if 'K' == x[-1]:
        if len(x) > 1:
            return float(x.replace('K', '')) * 1000
        return 1000.0
    if 'M' == x[-1]:
        if len(x) > 1:
            return float(x.replace('M', '')) * 1000000
        return 1000000.0
    if 'G' == x[-1]:
        return float(x.replace('G', '')) * 1000000000
    return x

def OpCounter(img, laneline, drivable, bbox, model, results_file, split):
    """get macs, params, flops, parameter count

    Args:
        img (torch.Tensor): Test data
        model (models): Test model
        results_file (pathlib): save resuslt
    """

    macs, params = profile(model, inputs=(img, laneline, drivable, bbox))  # ,verbose=False

    write_log(results_file, f"FLOPs: {macs}")
    write_log(results_file, f"MACs: {macs*2}")
    write_log(results_file, f"params: {params}")

    # stat(model, (img, laneline, drivable, bbox))
    # flops = FlopCountAnalysis(model, (img, laneline, drivable, bbox))
    # write_log(results_file, f"FLOPs: {flops.total()}")

    # write results to csv
    # def write_csv(fileName, table):
    #     parameter_data = table.split('\n')

    #     data = {}
    #     for i, index in enumerate(parameter_data[0].split('|')[1:-1], start=1):
    #         data[index.strip(' ')] = [value_to_float(line.split('|')[i].strip(' ')) for line in parameter_data[2:]]

    #     myvar = pd.DataFrame(data)
    #     myvar.to_csv(str(results_file).replace("results.txt",fileName))


    # parameter_table = parameter_count_table(model)
    # write_csv("parameter.csv", parameter_table)
    
    # flop_table = flop_count_table(flops)
    # write_csv("flop.csv", flop_table)