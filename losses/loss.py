import torch.nn.functional as F
import torch
from math import exp
from config import args
device = args.gpu


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    ret = ssim_map.mean()

    return 1-ret


def ssim_mri(fused, input_mri):
    ssim_mri = ssim(fused, input_mri)

    return ssim_mri


def ssim_spect(fused, input_spect):
    ssim_spect = ssim(fused, input_spect)

    return ssim_spect


def sobel_filter(img):
    # 使用Sobel滤波器计算边缘信息
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_x = sobel_x.cuda()
    sobel_y = sobel_y.cuda()
    
    # Sobel滤波器卷积到每个通道 (groups=channels)
    edge_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
    edge_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])

    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return edges


def calculate_loss_edges(Y, X):
    Ey = sobel_filter(Y)
    Ex = sobel_filter(X)

    loss_edges = torch.sum(torch.abs(Ey - Ex)) / (Y.shape[0] * Y.shape[2] * Y.shape[3])
    return loss_edges


def calculate_loss_pixels(Y, X):
    loss_pixels = torch.sum(torch.abs(Y - X)) / (Y.shape[0] * Y.shape[2] * Y.shape[3])
    return loss_pixels


def calculate_total_loss(Y, X):
    loss_edges = calculate_loss_edges(Y, X)
    loss_pixels = calculate_loss_pixels(Y, X)
    loss_total = loss_pixels + loss_edges
    return loss_total


def edge_mri(fused, input_mri):
    edge_mri = calculate_total_loss(fused, input_mri)

    return edge_mri


def edge_spect(fused, input_spect):
    edge_spect = calculate_total_loss(fused, input_spect)

    return edge_spect

