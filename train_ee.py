from tqdm import tqdm
import glob
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import cv2
import torch.nn.functional as F

from networks.SmoothEdgeExtra import SmoothEdgeExtra as net

class GetDataset(Dataset):
    def __init__(self, data_path1, data_path2):
        self.data_path1 = data_path1
        self.data_path2 = data_path2

    def __getitem__(self, index):
        edge = self.data_path1[index]
        target = self.data_path2[index]

        edge = cv2.imread(edge, 0)
        target = cv2.imread(target, 0)

        tran = transforms.ToTensor()
        edge = tran(edge)
        target = tran(target)
        return edge, target

    def __len__(self):
        return len(self.data_path1)


def train(train_loader, model, optimizer):
    loss_mean = []
    model.cuda()
    model.train()
    for i, (edge, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        edge = edge.cuda()
        target = target.cuda()
        out = model(edge)
        optimizer.zero_grad()
        loss = F.mse_loss(out, target)
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
    data_path1 = ''    # Function source images
    data_path2 = ''    # Using edge information from sobel as ground truth
    data_path1 = glob.glob(data_path1 + '*.png')
    data_path2 = glob.glob(data_path2 + '*.png')
    dataset = GetDataset(data_path1=data_path1, data_path2=data_path2)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=1,
                              pin_memory=True)
    model = net()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_plt = []
    for epoch in range(100):
        print('Epoch [%d/%d]' % (epoch + 1, 100))
        epoch_loss = train(train_loader, model, optimizer)
        print('Loss:%.5f' % epoch_loss)

        loss_plt.append(epoch_loss)
        strain_path = 'model_smooth_edge/loss.txt'
        loss_file = 'Epoch--' + str(epoch + 1) + '\t' + 'Loss:' + str(epoch_loss)
        with open(strain_path, 'a') as f:
            f.write(loss_file + '\r\n')

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), 'model_smooth_edge/model_{}.pth'.format(epoch + 1))

if __name__ == '__main__':
    main()
