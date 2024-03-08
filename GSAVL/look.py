from models.segment_anything import SamPredictor, sam_model_registry
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from test import NormReducer,Unsqueeze
import os
from model import VSL
from config import cfg
from datasets import load_color_mask_in_PIL_to_Tensor,get_v2_pallete
from torchvision import transforms
from sampler import samplers
import utils
import json
label2id={
    "background": 1,
    "accordion": 2,
    "airplane": 3,
    "axe": 4,
    "baby": 5,
    "bassoon": 6,
    "bell": 7,
    "bird": 8,
    "boat": 9,
    "boy": 10,
    "bus": 11,
    "car": 12,
    "cat": 13,
    "cello": 14,
    "clarinet": 15,
    "clipper": 16,
    "clock": 17,
    "dog": 18,
    "donkey": 19,
    "drum": 20,
    "duck": 21,
    "elephant": 22,
    "emergency-car": 23,
    "erhu": 24,
    "flute": 25,
    "frying-food": 26,
    "girl": 27,
    "goose": 28,
    "guitar": 29,
    "gun": 30,
    "guzheng": 31,
    "hair-dryer": 32,
    "handpan": 33,
    "harmonica": 34,
    "harp": 35,
    "helicopter": 36,
    "hen": 37,
    "horse": 38,
    "keyboard": 39,
    "leopard": 40,
    "lion": 41,
    "man": 42,
    "marimba": 43,
    "missile-rocket": 44,
    "motorcycle": 45,
    "mower": 46,
    "parrot": 47,
    "piano": 48,
    "pig": 49,
    "pipa": 50,
    "saw": 51,
    "saxophone": 52,
    "sheep": 53,
    "sitar": 54,
    "sorna": 55,
    "squirrel": 56,
    "tabla": 57,
    "tank": 58,
    "tiger": 59,
    "tractor": 60,
    "train": 61,
    "trombone": 62,
    "truck": 63,
    "trumpet": 64,
    "tuba": 65,
    "ukulele": 66,
    "utv": 67,
    "vacuum-cleaner": 68,
    "violin": 69,
    "wolf": 70,
    "woman": 71
}
id2label={}
for i in label2id.keys():
    label2id[i]=label2id[i]-1
    id2label[label2id[i]]=i
with open("label2id.json","w") as f:
     json.dump(label2id,f)

with open("id2label.json", "w") as f:
    json.dump(id2label, f)