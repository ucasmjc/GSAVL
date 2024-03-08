import json
from torch.optim import *
from sklearn import metrics
import time
import os
import shutil
import numpy as np
from PIL import Image
import logging
import random

import torch
def get_v2_pallete(num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete  # list, lenth is n_classes*3

    v2_pallete = _getpallete(num_cls)  # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)

    return v2_pallete


def color_mask_to_label(mask, v_pallete):
    mask_array = np.array(mask).astype('int32')
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    # pdb.set_trace() # there is only one '1' value for each pixel, run np.sum(semantic_map, axis=-1)
    label = np.argmax(semantic_map, axis=-1)
    return label


def load_color_mask_in_PIL_to_Tensor(path, v_pallete, mode='RGB'):
    color_mask_PIL = Image.open(path).convert(mode)
    # obtain semantic label
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label)  # [H, W]
    color_label = color_label.unsqueeze(0)
    binary_mask = (color_label != 0).float()
    # return color_label, binary_mask # both [1, H, W]
    return color_label,binary_mask


class Evaluator(object):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
        if thres:
            infer_map = np.zeros(gtmap.shape)
            infer_map[infer >= thres] = 1
        else:
            infer_map=infer
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap==0)))
        self.ciou.append(ciou)
        return ciou, np.sum(infer_map*gtmap), (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))

    def finalize_AUC(self):
        cious = [np.sum(np.array(self.ciou) >= 0.05*i) / len(self.ciou)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def finalize_AP50(self):
        ap50 = np.mean(np.array(self.ciou) >= 0.1)
        return ap50

    def finalize_cIoU(self):
        ciou = np.mean(np.array(self.ciou))
        return ciou

    def clear(self):
        self.ciou = []


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


def visualize(raw_image, boxes):
    import cv2
    boxes_img = np.uint8(raw_image.copy())[:, :, ::-1]

    for box in boxes:

        xmin,ymin,xmax,ymax = int(box[0]),int(box[1]),int(box[2]),int(box[3])

        cv2.rectangle(boxes_img[:, :, ::-1], (xmin, ymin), (xmax, ymax), (0,0,255), 1)

    return boxes_img[:, :, ::-1]


def build_optimizer_and_scheduler_adam(model, args):
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(optimizer_grouped_parameters, lr=args.init_lr)
    scheduler = None
    return optimizer, scheduler


def build_optimizer_and_scheduler_sgd(model, args):
    optimizer_grouped_parameters = model.parameters()
    optimizer = SGD(optimizer_grouped_parameters, lr=args.init_lr)
    scheduler = None
    return optimizer, scheduler


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode='w', encoding='utf-8') as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

def save_iou(iou_list, suffix, output_dir):
    # sorted iou
    sorted_iou = np.sort(iou_list).tolist()
    sorted_iou_indices = np.argsort(iou_list).tolist()
    file_iou = open(os.path.join(output_dir,"iou_test_{}.txt".format(suffix)),"w")
    for indice, value in zip(sorted_iou_indices, sorted_iou):
        line = str(indice) + ',' + str(value) + '\n'
        file_iou.write(line)
    file_iou.close()
def setup_logging(filename, resume=False):
    root_logger = logging.getLogger()

    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=filename, mode='a' if resume else 'w')

    root_logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    root_logger.addHandler(ch)
    root_logger.addHandler(fh)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class AverageMeter(object):

    def __init__(self, window=-1):
        self.window = window
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.max = -np.Inf

        if self.window > 0:
            self.val_arr = np.zeros(self.window)
            self.arr_idx = 0

    def update(self, val, n=1):
        self.val = val
        self.cnt += n
        self.max = max(self.max, val)

        if self.window > 0:
            self.val_arr[self.arr_idx] = val
            self.arr_idx = (self.arr_idx + 1) % self.window
            self.avg = self.val_arr.mean()
        else:
            self.sum += val * n
            self.avg = self.sum / self.cnt


class FrameSecondMeter(object):

    def __init__(self):
        self.st = time.time()
        self.fps = None
        self.ed = None
        self.frame_n = 0

    def add_frame_n(self, frame_n):
        self.frame_n += frame_n

    def end(self):
        self.ed = time.time()
        self.fps = self.frame_n / (self.ed - self.st)


def gct(f='l'):
    '''
    get current time
    :param f: 'l' for log, 'f' for file name
    :return: formatted time
    '''
    if f == 'l':
        return time.strftime('%m/%d %H:%M:%S', time.localtime(time.time()))
    elif f == 'f':
        return time.strftime('%m_%d_%H_%M', time.localtime(time.time()))


def save_scripts(path, scripts_to_save=None):
    if not os.path.exists(os.path.join(path, 'scripts')):
        os.makedirs(os.path.join(path, 'scripts'))

    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_path = os.path.join(path, 'scripts', script)
            try:
                shutil.copy(script, dst_path)
            except IOError:
                os.makedirs(os.path.dirname(dst_path))
                shutil.copy(script, dst_path)


def count_model_size(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


def load_image_in_PIL(path, mode='RGB'):
    img = Image.open(path)
    img.load()  # Very important for loading large image
    return img.convert(mode)


def print_mem(info=None):
    if info:
        print(info, end=' ')
    mem_allocated = round(torch.cuda.memory_allocated() / 1048576)
    mem_cached = round(torch.cuda.memory_cached() / 1048576)
    print(f'Mem allocated: {mem_allocated}MB, Mem cached: {mem_cached}MB')


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out