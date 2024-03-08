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
from datasets import load_image_in_PIL_to_Tensor,load_audio_from_pkl,load_spectrogram
from torchvision import transforms
from sampler import samplers
import utils
#设置图片路径，频谱图路径，使用的模型，采样方式和使用的checkpoint
image_path="example/1.jpg"
pkl_path="example/1.pkl"
audio_path=None
is_vggish=True
sampler='prob'
experiment="train_avs_gsavl"
img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
img = load_image_in_PIL_to_Tensor(image_path,
                                          transform=img_transform)
image=Image.open(image_path)
predictor = SamPredictor(
        sam_model_registry["vit_h"](checkpoint="models/segment_anything/checkpoint/sam_vit_h_4b8939.pth"))
predictor.set_image(np.array(image).astype("uint8"))

model_dir = os.path.join("checkpoints", experiment)

# Models
model = VSL(is_vggish=is_vggish)
from torchvision.models import resnet18
object_saliency_model = resnet18(pretrained=True)
object_saliency_model.avgpool = nn.Identity()
object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
)
ckp_fn = os.path.join(model_dir, 'best.pth')
if os.path.exists(ckp_fn):
        ckp = torch.load(ckp_fn, map_location='cpu')
model.load_state_dict({k.replace('module.', '') if k.replace('module.', '').startswith("backbone.") else k.replace('module.', 'backbone.'): ckp['model'][k] for k in ckp['model']})

if is_vggish:
        spec = load_audio_from_pkl(pkl_path)

else:
        audio_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0], std=[12.0])])
        spec = audio_transform(load_spectrogram(audio_path))
sampler=samplers(sampler)
img=img.unsqueeze(0)
spec=spec.unsqueeze(0)
origin_size=np.array(image).shape[0:2]
heatmap_av = model(img.float(), spec.float())[1].unsqueeze(1)
heatmap_av = F.interpolate(heatmap_av, size=origin_size, mode='bilinear', align_corners=True)
heatmap_av = heatmap_av.data.cpu().numpy()
img_feat = object_saliency_model(img)
heatmap_obj = F.interpolate(img_feat, size=origin_size, mode='bilinear', align_corners=True)
heatmap_obj = heatmap_obj.data.cpu().numpy()
pred_av = utils.normalize_img(heatmap_av[0, 0])
pred_obj = utils.normalize_img(heatmap_obj[0, 0])
pred_av_obj = utils.normalize_img(pred_av * 0.4 + pred_obj * (1 - 0.4))
point_av_obj=sampler(pred_av_obj)
points=np.stack([point_av_obj])
N=point_av_obj.shape[0]
masks, _, _ = predictor.predict(point_coords=points, point_labels=np.ones((1,N)), multimask_output=False)
pred=Image.fromarray((masks[0,0,:,:]* 255).astype('uint8'))
pred.save("example/result.png")

