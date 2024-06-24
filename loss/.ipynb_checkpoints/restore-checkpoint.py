import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
# from skimage.measure import compare_ssim as SSIM
# from skimage.measure import compare_psnr as PSNR
# local modules
from .PerceptualSimilarity import models


class perceptual_loss():
    def __init__(self, weight=1.0, net='alex', use_gpu=True, gpu_ids=[0]):
        """
        Wrapper for PerceptualSimilarity.models.PerceptualLoss
        """
        self.model = models.PerceptualLoss(net=net, use_gpu=use_gpu, gpu_ids=gpu_ids)
        self.weight = weight

    def __call__(self, pred, target, normalize=True):
        """
        pred and target are Tensors with shape N x C x H x W (C {1, 3})
        normalize scales images from [0, 1] to [-1, 1] (default: True)
        PerceptualLoss expects N x 3 x H x W.
        """
        assert pred.size() == target.size()

        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
            target = torch.cat([target, target, target], dim=1)
            dist = self.model.forward(pred, target, normalize=normalize)
        elif pred.shape[1] == 3:
            dist = self.model.forward(pred, target, normalize=normalize)
        else:
            num_ch = pred.shape[1]
            dist = 0
            for idx in range(num_ch):
                dist += self.model.forward(pred[:, idx].repeat(1, 3, 1, 1), target[:, idx].repeat(1, 3, 1, 1), normalize=normalize)
            dist /= num_ch

        return self.weight * dist.mean()


class ssim_loss():
    def __init__(self):
        self.ssim = SSIM

    def __call__(self, pred, tgt):
        """
        pred, tgt: torch.tensor, 1xNxHxW
        """
        assert pred.size() == tgt.size()
        pred = pred.squeeze().cpu().numpy()
        tgt = tgt.squeeze().cpu().numpy()

        if len(pred.shape) == 3:
            num_ch = pred.shape[0]
            loss = 0
            for idx in range(num_ch):
                loss += self.ssim(pred[idx], tgt[idx])
            loss /= num_ch
        else:
            loss = self.ssim(pred, tgt)

        return loss


class psnr_loss():
    def __init__(self):
        self.psnr = PSNR

    def __call__(self, pred, tgt):
        """
        pred, tgt: torch.tensor, 1xNxHxW
        """
        assert pred.size() == tgt.size()
        pred = pred.squeeze().cpu().numpy()
        tgt = tgt.squeeze().cpu().numpy()

        if len(pred.shape) == 3:
            num_ch = pred.shape[0]
            loss = 0
            for idx in range(num_ch):
                # data_range = max(tgt[idx].max()-tgt.min(), pred[idx].max()-pred[idx].min())
                data_range = tgt[idx].max()-tgt.min()
                loss += self.psnr(tgt[idx], pred[idx], data_range=data_range)
            loss /= num_ch
        else:
            loss = self.psnr(pred.clip(0, 1), tgt.clip(0, 1))

        # loss = self.psnr((tgt.squeeze().cpu().numpy()*255).astype(np.uint8), (pred.squeeze().cpu().numpy()*255).astype(np.uint8), data_range=255)

        return loss