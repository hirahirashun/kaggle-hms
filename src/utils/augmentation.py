import numpy as np
import torch


def mixup_data(x, y, y2, alpha=2.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    mixed_y2 = lam * y2 + (1 - lam) * y2[index, :]

    return mixed_x, mixed_y, mixed_y2, index, lam


def get_rand_bbox(x, lam):
    width = x.shape[1]
    height = x.shape[2]
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - lam)
    r_w = int(width * r_l)
    r_h = int(height * r_l)
    bb_x_1 = int(np.clip(r_x - r_w, 0, width))
    bb_y_1 = int(np.clip(r_y - r_h, 0, height))
    bb_x_2 = int(np.clip(r_x + r_w, 0, width))
    bb_y_2 = int(np.clip(r_y + r_h, 0, height))

    return bb_x_1, bb_y_1, bb_x_2, bb_y_2


def cutmix_data(x, y, y2, alpha=2.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    for i in range(x.shape[0]):
        bx1, by1, bx2, by2 = get_rand_bbox(x[i], lam)
        x[i, :, bx1:bx2, by1:by2] = x[index[i], :, bx1:bx2, by1:by2]

    y = lam * y + (1 - lam) * y[index, :]
    y2 = lam * y2 + (1 - lam) * y2[index, :]

    return x, y, y2, index, lam