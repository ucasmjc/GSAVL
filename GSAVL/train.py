import os
import argparse
import builtins
import time
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist
from config import cfg
from utils import setup_logging
import utils
from model import VSL
from datasets import get_vgg_test_dataset,get_vgg_train_dataset,get_avs_train_dataset,get_avs_test_dataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='ezvsl_vggss', help='experiment name (used for checkpointing and logging)')
    parser.add_argument('--trainset', default='vggss', type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='vggss', type=str, help='testset,(flickr or vggss)')
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-is_vggish', action='store_true')

    return parser.parse_args()



def main_worker(args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)

    model = VSL(is_vggish=args.is_vggish)

    FixSeed = 123
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    log_path=os.path.join(model_dir, 'log.txt')
    setup_logging(filename=log_path)
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.experiment_name))

    logger.info("==> Total params: %.2fM" % ( sum(p.numel() for p in model.parameters()) / 1e6))
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    print(model)


    optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(model, args)

    # Resume if possible
    start_epoch, best_cIoU, best_Auc = 0, 0., 0.
    if os.path.exists(os.path.join(model_dir, 'latest.pth')):
        ckp = torch.load(os.path.join(model_dir, 'latest.pth'), map_location='cuda:0')
        start_epoch, best_cIoU, best_Auc = ckp['epoch'], ckp['best_cIoU'], ckp['best_Auc']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        logger.info(f'loaded from {os.path.join(model_dir, "latest.pth")}')
    if args.trainset=="vggss":
        # Dataloaders
        traindataset = get_vgg_train_dataset(args)
        train_loader = torch.utils.data.DataLoader(
            traindataset, batch_size=args.batch_size, shuffle=True,
            pin_memory=False,  drop_last=True)
        testdataset = get_vgg_test_dataset(args)
        test_loader = torch.utils.data.DataLoader(
            testdataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
    elif args.trainset=="avs":
        # Dataloaders
        traindataset = get_avs_train_dataset(args)
        train_loader = torch.utils.data.DataLoader(
            traindataset, batch_size=args.batch_size, shuffle=True,
            pin_memory=False, drop_last=True)

        testdataset = get_avs_test_dataset(args)
        test_loader = torch.utils.data.DataLoader(
            testdataset, batch_size=1, shuffle=False,pin_memory=False, drop_last=False)

    logger.info("Loaded dataloader.")

    cIoU, auc = validate(test_loader, model, args)
    validate_log=f'cIoU (epoch {start_epoch}): {cIoU} \n'+f'AUC (epoch {start_epoch}): {auc}\n'+f'best_cIoU: {best_cIoU}\n'+f'best_Auc: {best_Auc}\n'
    logger.info(validate_log)
    for epoch in range(start_epoch, args.epochs):
        # Train
        train(train_loader, model, optimizer, epoch, args)
        # Evaluate
        cIoU, auc = validate(test_loader, model, args)
        train_log=f'cIoU (epoch {epoch + 1}): {cIoU}\n'+f'AUC (epoch {epoch + 1}): {auc}\n'+f'best_cIoU: {best_cIoU}\n'+f'best_Auc: {best_Auc}\n'
        logger.info(train_log)
        # Checkpoint
        ckp = {'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'epoch': epoch + 1,
          'best_cIoU': best_cIoU,
          'best_Auc': best_Auc}
        torch.save(ckp, os.path.join(model_dir, 'latest.pth'))
        logger.info(f"Model saved to {model_dir}")
        if cIoU >= best_cIoU:
            best_cIoU, best_Auc = cIoU, auc
            torch.save(ckp, os.path.join(model_dir, 'best.pth'))
def train(train_loader, model, optimizer, epoch, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_mtr = AverageMeter('Loss', ':.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_mtr],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i, (image, spec) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        loss, _ = model(image.float(), spec.float())
        loss_mtr.update(loss.item(), image.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 or i == len(train_loader) - 1:
            progress.display(i)
        del loss

def validate(test_loader, model, args):
    model.train(False)
    evaluator = utils.Evaluator()
    if args.testset == "vgg_test" or args.testset == "vgg_val":
        for step, (image, spec, bboxes, _) in enumerate(test_loader):
            if torch.cuda.is_available():
                spec = spec.cuda(args.gpu, non_blocking=True)
                image = image.cuda(args.gpu, non_blocking=True)

            avl_map = model(image.float(), spec.float())[1].unsqueeze(1)
            avl_map = F.interpolate(avl_map, size=(224, 224), mode='bicubic', align_corners=False)
            avl_map = avl_map.data.cpu().numpy()

            for i in range(spec.shape[0]):
                pred = utils.normalize_img(avl_map[i, 0])
                gt_map = bboxes['gt_map'].data.squeeze().cpu().numpy()
                thr = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
                evaluator.cal_CIOU(pred, gt_map, thr)
    elif args.testset=="avs":
        for step, (image, spec, mask, origin) in enumerate(test_loader):
            if torch.cuda.is_available():
                spec = spec.cuda(args.gpu, non_blocking=True)
                image = image.cuda(args.gpu, non_blocking=True)
            avl_map = model(image.float(), spec.float())[1].unsqueeze(1)
            avl_map = F.interpolate(avl_map, size=(mask.shape[-2],mask.shape[-1]), mode='bicubic', align_corners=False)
            avl_map = avl_map.data.cpu().numpy()
            pred = utils.normalize_img(avl_map)
            gt_map = mask.cpu().numpy()
            thr = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] *0.5)]
            evaluator.cal_CIOU(pred, gt_map, thr)
    cIoU = evaluator.finalize_cIoU()
    AUC = evaluator.finalize_AUC()
    return cIoU, AUC


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main_worker(get_arguments())

