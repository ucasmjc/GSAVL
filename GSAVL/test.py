import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as f
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse
from config import cfg
from utils import setup_logging
import logging
from models.ezvsl import EZVSL
from models.GSAVL import GSAVL
from models.segment_anything import SamPredictor, sam_model_registry
from datasets import get_vgg_test_dataset,get_avs_test_dataset, inverse_normalize
from sampler import samplers
from model import VSL
import cv2
from PIL import Image

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='current', help='experiment name (experiment folder set to "args.model_dir/args.experiment_name)"')
    parser.add_argument('--testset', default='flickr', type=str, help='testset (vgg_test or avs)')
    parser.add_argument('--save_visualizations', action='store_true', help='Set to store all VSL visualizations (saved in viz directory within experiment folder)')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('-is_vggish', action='store_true')
    parser.add_argument('-sam_used', action='store_true')
    parser.add_argument('-obj_used', action='store_true')
    parser.add_argument('--sampler_type', default='naive')
    return parser.parse_args()


def main(args):
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    viz_dir = os.path.join(model_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    log_path = os.path.join(model_dir, 'log.txt')
    setup_logging(filename=log_path)
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.experiment_name))
    # Models
    model = VSL(is_vggish=args.is_vggish).to("cpu")
    from torchvision.models import resnet18
    object_saliency_model = resnet18(pretrained=True)
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        object_saliency_model.cuda(args.gpu)

    ckp_fn = os.path.join(model_dir, 'best.pth')
    if os.path.exists(ckp_fn):
        ckp = torch.load(ckp_fn, map_location='cpu')
        model.load_state_dict({k.replace('module.', '') if k.replace('module.', '').startswith("backbone.") else k.replace('module.', 'backbone.'): ckp['model'][k] for k in ckp['model']})
        logger.info(f'loaded from {os.path.join(model_dir, "best.pth")}')
    else:
        print(f"Checkpoint not found: {ckp_fn}")

    if args.testset == "vgg_test" or args.testset == "vgg_val":
        # Dataloaders
        testdataset = get_vgg_test_dataset(args)
        testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False)
    elif args.testset=="avs":
        # Dataloaders
        testdataset = get_avs_test_dataset(args)
        testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False)
    print("Loaded dataloader.")
    validate(testdataloader, model, object_saliency_model, viz_dir, args)


@torch.no_grad()
def validate(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args):
    audio_visual_model.train(False)
    object_saliency_model.train(False)

    evaluator_av = utils.Evaluator()
    evaluator_obj = utils.Evaluator()
    evaluator_av_obj = utils.Evaluator()
    predictor = SamPredictor(
        sam_model_registry["vit_h"](checkpoint="models/segment_anything/checkpoint/sam_vit_h_4b8939.pth"))
    for step, (image, spec, label, origin) in enumerate(testdataloader):
        origin_size = (224,224) if args.testset != "avs" else label.shape[-2:]
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        # Compute S_AVL
        heatmap_av = audio_visual_model(image.float(), spec.float())[1].unsqueeze(1)
        heatmap_av = F.interpolate(heatmap_av, size=origin_size, mode='bilinear', align_corners=True)
        heatmap_av = heatmap_av.data.cpu().numpy()
        if args.obj_used==True:
            logging.info("obj is used.")
            img_feat = object_saliency_model(image)
            heatmap_obj = F.interpolate(img_feat, size=origin_size, mode='bilinear', align_corners=True)
            heatmap_obj = heatmap_obj.data.cpu().numpy()
        else:
            heatmap_obj=heatmap_av
        alpha=0.4
        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            pred_av = utils.normalize_img(heatmap_av[i, 0])
            pred_obj = utils.normalize_img(heatmap_obj[i, 0])
            pred_av_obj = utils.normalize_img(pred_av * alpha + pred_obj * (1 - alpha))


            if args.testset == "vgg_test" or args.testset == "vgg_val":
                gt_map = label['gt_map'].squeeze().cpu().numpy()
                beta=0.5
            elif args.testset == "avs":
                gt_map = label.squeeze().cpu().numpy().astype("uint8")
                name = origin[1][0]
                origin=origin[0].squeeze().cpu().numpy().astype("uint8")
            if args.sam_used==True:
                logging.info("sam is used.")
                sampler=samplers(args.sampler_type)
                points_av=sampler(pred_av)
                point_obj=sampler(pred_obj)
                point_av_obj=sampler(pred_av_obj)
                predictor.set_image(origin)
                points=np.stack([points_av,point_obj,point_av_obj])
                N=points_av.shape[0]
                # 返回mask，iou预测和模型的直接输出（未经过后处理裁剪，256*256)，numpy格式,B*1*H*W
                masks, _, _ = predictor.predict(point_coords=points, point_labels=np.ones((3,N)), multimask_output=False)
                #image = Image.fromarray(origin)
                #image.save(os.path.join(viz_dir, f'{step}_img.jpg'))
                #mask = Image.fromarray((gt_map * 255).astype('uint8'))
                #mask.save(os.path.join(viz_dir, f'{step}_gt.png'))
                #mask=Image.fromarray((masks[0,0,:,:]*255).astype('uint8'))
                #mask.save(os.path.join(viz_dir, f'{step}_av.png'))
                #mask = Image.fromarray((masks[1, 0, :, :] * 255).astype('uint8'))
                #mask.save(os.path.join(viz_dir, f'{step}_obj.png'))
                #mask = Image.fromarray((masks[2, 0, :, :] * 255).astype('uint8'))
                #mask.save(os.path.join(viz_dir, f'{step}_avobj.png'))
                iou_av, _, _ = evaluator_av.cal_CIOU(masks[0, 0, :, :], gt_map, None)
                iou_obj, _, _ = evaluator_obj.cal_CIOU(masks[1, 0, :, :], gt_map, None)
                iou_avobj, _, _ = evaluator_av_obj.cal_CIOU(masks[2, 0, :, :], gt_map, None)
            else:
                thr_av = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1] * beta)]
                iou_av, _, _=evaluator_av.cal_CIOU(pred_av, gt_map, thr_av)

                thr_obj = np.sort(pred_obj.flatten())[int(pred_obj.shape[0] * pred_obj.shape[1] * beta)]
                iou_obj, _, _ =evaluator_obj.cal_CIOU(pred_obj, gt_map, thr_obj)

                thr_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * beta)]
                iou_avobj, _, _ =evaluator_av_obj.cal_CIOU(pred_av_obj, gt_map, thr_av_obj)

            if args.save_visualizations:
                thr_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * beta)]
                pred_av_obj[pred_av_obj>thr_av_obj]=255
                pred_av_obj[pred_av_obj < thr_av_obj] = 0
                mask = Image.fromarray((pred_av_obj).astype('uint8'))
                mask.save(os.path.join(viz_dir, f'{step}_avl_av_obj.jpg'))

        logging.info(f'{step+1}/{len(testdataloader)}:{iou_avobj:.2f}')
    def compute_stats(eval):
        mAP = eval.finalize_AP50()
        ciou = eval.finalize_cIoU()
        auc = eval.finalize_AUC()
        return mAP, ciou, auc
    logging.info('AV_Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av_obj)))


class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


if __name__ == "__main__":
    main(get_arguments())

