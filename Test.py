import os
import torch
import cv2
import copy
import numpy as np
import time
from torch import nn

from config import args
import torchvision.transforms as transforms
from networks.net import MODEL as net
from networks.SmoothEdgeExtra import SmoothEdgeExtra

device = args.gpu
model = net()
model.eval()
model = model.to(device)
model.load_state_dict(torch.load('model_save/model_100.pth'))

model1 = SmoothEdgeExtra()
model1.eval()
model1 = model1.to(device)
model1.load_state_dict(torch.load('model_smooth_edge/model_100.pth'))


def MultiFusion():
    path = 'dataset/test/MRI-SPECT/MRI/'
    files = os.listdir(path)
    Time = []
    for file in files:
        with torch.no_grad():
            path1 = os.path.join('dataset/test/MRI-SPECT/SPECT', file)
            path2 = os.path.join('dataset/test/MRI-SPECT/MRI', file)
            img_bgr_cv = cv2.imread(path1, 1)
            img_gray_y_cv = cv2.imread(path2, 0)

            img_yuv_cv = cv2.cvtColor(img_bgr_cv, cv2.COLOR_BGR2YUV)
            img_y_yuv_cv = img_yuv_cv[:, :, 0]

            tran = transforms.ToTensor()

            img_y_yuv_cv = tran(img_y_yuv_cv)
            img_gray_y_cv = tran(img_gray_y_cv)

            img_y_yuv_cv = img_y_yuv_cv.to(device)
            img_gray_y_cv = img_gray_y_cv.to(device)

            img_y_yuv_cv = img_y_yuv_cv.unsqueeze(0)
            img_gray_y_cv = img_gray_y_cv.unsqueeze(0)

            model.eval()
            tic = time.time()
            img_y_yuv_cv = model1(img_y_yuv_cv)
            out = model(img_y_yuv_cv, img_gray_y_cv)

            out_y = ((out[0][0]).detach().cpu().numpy() * 255).astype(np.uint8)
            result_yuv = copy.deepcopy(img_yuv_cv)
            result_yuv[:, :, 0] = out_y
            result_bgr = cv2.cvtColor(result_yuv, cv2.COLOR_YUV2BGR)
            cv2.imwrite('result/EFCNet/{}.png'.format(file.split('.')[0]), result_bgr)
        del img_gray_y_cv
        torch.cuda.empty_cache()
        toc = time.time()
        Time.append(toc - tic)
    print('Time:', np.mean(Time))
    

if __name__ == '__main__':
    MultiFusion()
