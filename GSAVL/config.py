from easydict import EasyDict as edict

cfg = edict()
cfg.VGGISH_BATCH_SIZE = 4
cfg.LAMBDA_1 = 50
cfg.TRAIN = edict()
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "models/vggish/vggish-10086976.pth"
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "models/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.VGGISHDEVICE="cpu"
###############################
# DATA
cfg.DATA = edict()
cfg.DATA.AVS_CSV = "/Users/24967\Desktop\AVS\metadata.csv"
cfg.DATA.AVS_PKL = "/Users/24967\Desktop\AVS/v1s_pkl/"
cfg.DATA.AVS_DATA = "/Users/24967\Desktop\AVS/v1s"
cfg.DATA.VGGSOUND_DATA = "C:/Users/24967\Desktop/vggsound/"
cfg.DATA.DIR_MASK = "/home/data/AVSBench_data/Multi-sources/ms3_data/gt_masks"
cfg.DATA.IMG_SIZE = (224, 224)
