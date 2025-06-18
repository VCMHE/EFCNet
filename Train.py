from tqdm import tqdm
import glob
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch.nn.functional as F
from losses.loss import ssim_spect, ssim_mri, edge_spect, edge_mri
from networks.net import MODEL as net
from networks.SmoothEdgeExtra import SmoothEdgeExtra
from config import args
device = args.gpu

class GetDataset(Dataset):
    def __init__(self, data_path1, data_path2):
        self.data_path1 = data_path1
        self.data_path2 = data_path2

    def __getitem__(self, index):
        spect = self.data_path1[index]
        mri = self.data_path2[index]

        spect = cv2.imread(spect, 0)
        mri = cv2.imread(mri, 0)

        tran = transforms.ToTensor()
        spect = tran(spect)
        mri = tran(mri)

        return spect, mri

    def __len__(self):
        return len(self.data_path1)


def train(train_loader, model, optimizer, model1):
    loss_mean = []
    model.to(device)
    model1.to(device)
    model.train()
    model1.eval()
    for i, (spect, mri) in tqdm(enumerate(train_loader),  total=len(train_loader)):
        spect = spect.to(device)
        mri = mri.to(device)
        spect = model1(spect)
        out = model(spect, mri)
        s_spect = ssim_spect(out, spect)
        s_mri = ssim_mri(out, mri)
        e_spect = edge_spect(out, spect)
        e_mri = edge_mri(out, mri)
        loss = s_spect + s_mri + e_spect + e_mri
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_mean.append(loss.item())
    epoch_loss = sum(loss_mean) / len(loss_mean)
    return epoch_loss


def setup_seed(seed=302):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed()
    data_path1 = 'dataset/train/SPECT/'
    data_path2 = 'dataset/train/MRI/'
    data_path1 = glob.glob(data_path1 + '*.png')
    data_path2 = glob.glob(data_path2 + '*.png')
    dataset = GetDataset(data_path1=data_path1, data_path2=data_path2)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=1,
                              pin_memory=True)
    model = net()
    model1 = SmoothEdgeExtra()
    model1.load_state_dict(torch.load('networks/model_100.pth'))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_plt = []
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))
        epoch_loss = train(train_loader, model, optimizer, model1)
        print('Loss:%.5f' % epoch_loss)

        loss_plt.append(epoch_loss)
        strain_path = 'model_save/loss.txt'
        loss_file = 'Epoch--' + str(epoch + 1) + '\t' + 'Loss:' + str(epoch_loss)
        with open(strain_path, 'a') as f:
            f.write(loss_file + '\r\n')
            
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), 'model_save/model_{}.pth'.format(epoch+1))
    plt.figure()
    x = range(0, args.epochs)  # x和y的维度要一样
    y = loss_plt
    plt.plot(x, y, 'r-')  # 设置输出样式
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('Light/loss.png')  # 保存训练损失曲线图片
    plt.show()  # 显示曲线


if __name__ == '__main__':
    main()
