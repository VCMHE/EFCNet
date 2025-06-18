import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='cuda:0', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
args = parser.parse_args()
