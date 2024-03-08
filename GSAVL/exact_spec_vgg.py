from models.segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as f
from test import NormReducer,Unsqueeze
import os
from model import VSL
from config import cfg
from datasets import load_image_in_PIL_to_Tensor,load_audio_from_pkl,load_spectrogram,open_audio_av,load_audio_av
from models.vggish.vggish_input import waveform_to_examples
from torchvision import transforms
from sampler import samplers
import utils
import pickle

audio_path = cfg.DATA.VGGSOUND_DATA+"test_aud/"
image_path = cfg.DATA.VGGSOUND_DATA+"test_img/"
save_path=cfg.DATA.VGGSOUND_DATA+"test_pkl/"
# List directory
audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
image_files = {f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))}
avail_files = audio_files.intersection(image_files)
print(f"{len(avail_files)} available files")
subset = set(open(f"metadata/new_vggss_test.txt").read().splitlines())
avail_files = avail_files.intersection(subset)
avail_files = sorted(list(avail_files))
audio_files = sorted([dt + '.wav' for dt in avail_files])
image_files = sorted([dt for dt in avail_files])
dur=3.
for i in audio_files:
    audio_ctr = open_audio_av(audio_path+i)
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
    spec=waveform_to_examples(audio,samplerate)
    lm_save_path=save_path+i.split('.wav')[0]+".pkl"
    with open(lm_save_path, "wb") as fw:
        pickle.dump(spec, fw)


