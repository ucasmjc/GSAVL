import torch
from torch import nn
import torch.nn.functional as F
from models.ezvsl import EZVSL
from models.GSAVL import GSAVL


class VSL(nn.Module):
    def __init__(self, is_vggish=True):
        super(VSL, self).__init__()
        if is_vggish == True:
            self.backbone = GSAVL(512)
        else:
            self.backbone = EZVSL(512)
        self.tau = 0.03

    def loss(self, img, aud):
        loss, logits = self.max_xmil_loss(img, aud)

        return loss, logits

    def max_xmil_loss(self, img, aud):
        B = img.shape[0]
        Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / self.tau
        logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
        labels = torch.arange(B).long().to(img.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.permute(1, 0), labels)
        return loss, Slogits

    def forward(self, image, audio):
        if len(audio.shape) == 5:
            audio = audio.flatten(0, 1)
            img, aud = self.backbone(image, audio)
            aud = aud.unflatten(0, (img.shape[0], -1))
            aud = torch.mean(aud, dim=1)
        else:
            img, aud = self.backbone(image, audio)

        loss, logits = self.max_xmil_loss(img, aud)  # b*b*h*w
        with torch.no_grad():
            B = img.shape[0]
            Savl = logits[torch.arange(B), torch.arange(B)]  # 正确图-音对的mask，B*H*W
        return loss, Savl




