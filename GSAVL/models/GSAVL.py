
from torch import nn

from torchvision.models import resnet18
from models.vggish import vggish
from easydict import EasyDict as edict
from config import cfg
class GSAVL(nn.Module):
    def __init__(self, dim):
        super(GSAVL, self).__init__()
        # Vision model
        self.imgnet = resnet18(pretrained=True)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()
        self.img_proj = nn.Conv2d(512, dim, kernel_size=(1, 1))

        # Audio model

        self.audnet = vggish.VGGish(cfg)
        self.aud_proj = nn.Linear(128, dim)

        # Initialize weights (except pretrained visual model)
        for net in [self.audnet, self.img_proj, self.aud_proj]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)



    def forward(self, image, audio):
        # Image
        img = self.imgnet(image).unflatten(1, (512, 7, 7))
        img = self.img_proj(img)
        img = nn.functional.normalize(img, dim=1)

        # Audio
        aud = self.audnet(audio)
        aud = self.aud_proj(aud)
        aud = nn.functional.normalize(aud, dim=1)

        return img,aud