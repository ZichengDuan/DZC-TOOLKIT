from test_loader.sentryDataset import *
from test_loader.rgbdDataset import *
from test_loader.rgbDataset import *
from test_loader.dataset import FrameDataset
from test_loader.visualizer import Viualizer
import torch
import argparse

def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    dataset = args.dataset
    dataroot = args.root

    if dataset == "sentry":
        base = sentryDataset(os.path.join(dataroot, dataset), worldgrid_shape=[449, 800])
        train_set =  FrameDataset(base, grid_reduce=4, img_reduce=4, train_ratio=args.train_ratio, train=True)
        test_set =  FrameDataset(base, grid_reduce=4, img_reduce=4, train_ratio=args.train_ratio, train=False)
    if dataset == "rgb" or "rgbd":
        base = rgbDataset(os.path.join(dataroot, dataset))
        train_set = FrameDataset(base, train_ratio=args.train_ratio, train=True)
        test_set = FrameDataset(base, train_ratio=args.train_ratio, train=False)
    if dataset == "rgbd":
        base = rgbdDataset(os.path.join(dataroot, dataset))
        train_set = FrameDataset(base, train_ratio=args.train_ratio, train=True)
        test_set = FrameDataset(base, train_ratio=args.train_ratio, train=False)

    # normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #
    # train_trans = T.Compose([T.ToTensor(), normalize])
    # test_trans = T.Compose([T.ToTensor(), normalize])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True, drop_last=True)

    for batch_idx, data in enumerate(train_loader):
        imgs, depth = data
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataloader API')
    parser.add_argument('-d', '--dataset', type=str, default='rgbd', choices=['sentry', 'rgb', 'rgbd'])
    parser.add_argument('-p', '--root', type=str, default="/home/dzc/Data/3in1")
    parser.add_argument('-s', '--seed', type=int, default=7)
    parser.add_argument('-t', '--train_ratio', type=int, default=0.01)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-w', '--num_workers', type=int, default=8)
    args = parser.parse_args()
    # main(args)

    vis = Viualizer("/home/dzc/Data/3in1", "sentry")
    vis.visualize(10)
    pass