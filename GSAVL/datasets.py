import os
import csv
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
from audio_io import load_audio_av, open_audio_av
import pandas as pd
import pickle
from config import cfg
from PIL import Image
from torchvision import transforms
from utils import get_v2_pallete,load_color_mask_in_PIL_to_Tensor

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_spectrogram(path, dur=3.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur) / 2 - dur / 2, 0)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram


def load_audio_from_pkl(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()  # [5, 1, 96, 64]
    return audio_log_mel


def load_all_bboxes():
    gt_bboxes = {}
    with open('metadata/vggss.json') as json_file:
        annotations = json.load(json_file)
    for annotation in annotations:
        bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in annotation['bbox']]
        file = annotation['file'].strip()
        new_line = ""
        splited = file.split("_")
        for i in splited[:-1]:
            new_line += i + "_"
        num = splited[-1].lstrip('0')
        num = num if len(num) > 0 else "0"
        new_line += num
        gt_bboxes[new_line] = bboxes

    return gt_bboxes


def bbox2gtmap(bboxes):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp
    gt_map[gt_map > 0] = 1
    return gt_map


class VGGDataset(Dataset):
    def __init__(self, image_files, audio_files, image_path, audio_path, split,audio_dur=3., is_vggish=False,image_transform=None,
                 audio_transform=None, all_bboxes=None):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path
        self.audio_dur = audio_dur

        self.audio_files = audio_files
        self.image_files = image_files
        self.all_bboxes = all_bboxes
        self.split=split
        self.image_transform = image_transform
        self.audio_transform = audio_transform
        self.is_vggish=is_vggish

    def getitem(self, idx):
        file_id = self.image_files[idx]

        # Image
        img_fn = os.path.join(self.image_path, file_id, "frame_00001.jpg")
        frame = self.image_transform(load_image_in_PIL_to_Tensor(img_fn))
        bboxes = {}
        if self.all_bboxes is not None:
            bboxes['bboxes'] = self.all_bboxes[file_id]
            bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file_id])
        if self.is_vggish:
            audio_lm_path = os.path.join(cfg.DATA.VGGSOUND_DATA,self.split+"_pkl", file_id + '.pkl')
            audio_log_mel = load_audio_from_pkl(audio_lm_path)
            if self.split == 'train':
                return frame, audio_log_mel
            else:
                return frame, audio_log_mel, bboxes, file_id
        else:
            # Audio
            audio_fn = os.path.join(self.audio_path, self.audio_files[idx])
            spectrogram = self.audio_transform(load_spectrogram(audio_fn))
            if self.split == 'train':
                return frame, spectrogram
            else:
                return frame, spectrogram, bboxes, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])


class AVSS4Dataset(Dataset):
    """Dataset for single sound source segmentation"""

    def __init__(self, split='train', is_vggish=True):
        super(AVSS4Dataset, self).__init__()
        self.split = split
        df_all = pd.read_csv(cfg.DATA.AVS_CSV, sep=',')
        subset = set(open(f"metadata/new_avs_" + split + ".txt").read().splitlines())
        self.df_split = df_all[df_all['uid'].isin(subset)]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.data_root = cfg.DATA.AVS_DATA
        self.pkl_root = cfg.DATA.AVS_PKL
        self.is_vggish = is_vggish
        self.pallete=get_v2_pallete(num_cls=71)

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video.iloc[1]
        data_path = os.path.join(self.data_root, video_name)
        mask_base_path = os.path.join(data_path, "labels_rgb")
        origin_image=np.array(Image.open(os.path.join(data_path, "frames/0.jpg"))).astype("uint8")
        img = load_image_in_PIL_to_Tensor(os.path.join(data_path, "frames/0.jpg"),
                                          transform=self.img_transform)

        mask = load_color_mask_in_PIL_to_Tensor(os.path.join(mask_base_path, "0.png"),self.pallete)
        if self.is_vggish:
            audio_lm_path = os.path.join(self.pkl_root, video_name + '.pkl')
            audio_log_mel = load_audio_from_pkl(audio_lm_path)
            # audio_lm_tensor = torch.from_numpy(audio_log_mel)

            if self.split == 'train':
                return img, audio_log_mel
            else:
                return img, audio_log_mel, mask,(origin_image,video_name)
        else:
            audio_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0], std=[12.0])])

            audio_fn = os.path.join(self.data_root, video_name, "audio.wav")
            spectrogram = audio_transform(load_spectrogram(audio_fn))

            if self.split == 'train':
                return img, spectrogram
            else:
                return img, spectrogram, mask, (origin_image,video_name)

    def __len__(self):
        return len(self.df_split)


def get_avs_train_dataset(args):
    return AVSS4Dataset(split='train', is_vggish=args.is_vggish)


def get_avs_test_dataset(args):
    return AVSS4Dataset(split='test', is_vggish=args.is_vggish)


def get_vgg_train_dataset(args):
    audio_path = cfg.DATA.VGGSOUND_DATA+"train_aud/"
    image_path = cfg.DATA.VGGSOUND_DATA+"train_img/"

    # List directory
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
    image_files = {f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))}
    avail_files = audio_files.intersection(image_files)
    print(f"{len(avail_files)} available files")

    subset = set(open(f"metadata/new_vggss_10k.txt").read().splitlines())
    avail_files = avail_files.intersection(subset)
    print(f"{len(avail_files)} valid subset files")
    avail_files = sorted(list(avail_files))
    audio_files = sorted([dt + '.wav' for dt in avail_files])
    image_files = sorted([dt for dt in avail_files])

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.1), Image.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[12.0])])

    return VGGDataset(
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        split="train",
        is_vggish=args.is_vggish
    )


def get_vgg_test_dataset(args):
    audio_path = cfg.DATA.VGGSOUND_DATA+'test_aud/'
    image_path = cfg.DATA.VGGSOUND_DATA+'test_img/'

    if args.testset == 'vgg_test':
        testtxt = 'metadata/new_vggss_test.txt'
    elif args.testset == 'vgg_val':
        testtxt = 'metadata/new_vggss_val.txt'
    else:
        raise NotImplementedError

    #  Retrieve list of audio and video files
    with open(testtxt, 'r') as file:
        testset = set(file.read().splitlines())

    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    image_files = {f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))}
    avail_files = audio_files.intersection(image_files)
    testset = testset.intersection(avail_files)
    avail_files = sorted(list(testset))
    audio_files = sorted([dt + '.wav' for dt in avail_files])
    image_files = sorted([dt for dt in avail_files])

    # Bounding boxes
    all_bboxes = load_all_bboxes()

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return VGGDataset(
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        all_bboxes=all_bboxes,
        split="test",
        is_vggish=args.is_vggish
    )


def inverse_normalize(tensor):
    inverse_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    inverse_std = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor



