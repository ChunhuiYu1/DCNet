import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def EGloss(predicted_images: torch.Tensor, real_images: torch.Tensor):
    conv_kernel = torch.ones(1, 1, 3, 3).to(real_images.device)
    weights = F.conv2d(real_images, conv_kernel, padding=1)
    real_images_cpu = real_images.cpu()
    weights_cpu = weights.cpu()
    weighted_weights = torch.tensor(
        np.where(real_images_cpu == 0, np.where(weights_cpu == 0, 1, 2), np.where(weights_cpu == 9, 1, 2))).to(
        real_images.device)
    cross_entropy = nn.BCELoss(weighted_weights)(predicted_images, real_images)
    return cross_entropy
