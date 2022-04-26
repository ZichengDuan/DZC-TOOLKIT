import argparse
import os
import cv2
import kornia
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import torch

import torchvision.transforms as T
from misc.datasets import oftFrameDataset, Robomaster_1_dataset, MultiviewX
intrinsic_camera_matrix_filenames = ["intri_left.xml", "intri_right.xml"]
extrinsic_camera_matrix_filenames = ["extri_left.xml", "extri_right.xml"]

def get_worldgrid_from_worldcoord(self, world_coord):
    coord_x, coord_y = world_coord
    grid_x = coord_x
    grid_y = coord_y
    return np.array([grid_x, grid_y], dtype=int)


def get_worldcoord_from_worldgrid(self, worldgrid):
    # datasets default unit: centimeter & origin: (-300,-900)
    grid_x, grid_y = worldgrid
    coord_x = grid_x
    coord_y = grid_y
    return np.array([coord_x, coord_y])

class DataProcessor():
    def __init__(self, dataset, data_dir):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.data_dir = data_dir
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        # calculate the
        self.temp_extrin = np.array([[ 0.6454993,  -0.74890889, -0.14988703, 192.54673737 / 10],
                                     [-0.51629561, -0.28324909, -0.80821334, 1456.81186847 / 10],
                                     [ 0.56282279,  0.59908716, -0.56949546, 1065.88383233/ 10]])



        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           # self.temp_extrin,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]

    def testPrint(self):
        img = Image.open("/home/dzc/Data/1carreal/pts/000131.jpg").convert('RGB')
        img_left = ToTensor()(img)
        # proj_mat = self.proj_mats[0].repeat([1, 1, 1]).float()
        # world_left = kornia.warp_perspective(img_left.unsqueeze(0), proj_mat, self.reducedgrid_shape)

        test_img_left = kornia.warp_perspective(img_left , torch.tensor(
            np.linalg.inv(self.proj_mats[1].repeat([1, 1, 1]).float())), (480, 640))
        test_img_left = test_img_left[0, :].numpy().transpose([1, 2, 0])
        test_img_left = Image.fromarray((test_img_left * 255).astype('uint8'))
        test_img_left.save("test_print.jpg")


    def cropImage(self, root, imgFolder, isleft=True, id=-1):
        filenames = sorted(os.listdir(imgFolder + "left1"))
        print(filenames)
        for idx, filename in enumerate(filenames):
            print(idx)
            # 读取图片
            if id == -1 or idx in id :
                img1 = Image.open(imgFolder + "left1/%s" % filename).convert('RGB')
                img2 = Image.open(imgFolder + "right2/%s" % filename).convert('RGB')

                img1 = ToTensor()(img1)
                img2 = ToTensor()(img2)

                if isleft == 1:
                    left_proj_mat = self.proj_mats[0].repeat([1, 1, 1]).float()
                elif isleft == 2:
                    right_proj_mat = self.proj_mats[1].repeat([1, 1, 1]).float()
                else:
                    left_proj_mat = self.proj_mats[0].repeat([1, 1, 1]).float()
                    right_proj_mat = self.proj_mats[1].repeat([1, 1, 1]).float()
                # 模拟矩阵不准
                left_proj_mat = torch.matmul(left_proj_mat, torch.tensor([[[0.989,0.,0.9988], [0, 1, 0.], [0., 0., 1]]]))
                right_proj_mat = torch.matmul(right_proj_mat, torch.tensor([[[1, 0.,0.], [0, 1.1, 0.], [0., 0., 0.978]]]))

                world1 = kornia.geometry.warp_perspective(img1.unsqueeze(0), left_proj_mat, self.reducedgrid_shape)
                world2 = kornia.geometry.warp_perspective(img2.unsqueeze(0), right_proj_mat, self.reducedgrid_shape)

                world1 = kornia.geometry.vflip(world1)
                world2 = kornia.geometry.vflip(world2)

                # ---------------------处理tensor化的图片，回到原图-------------------
                world1 = world1[0, :].numpy().transpose([1, 2, 0])
                world1 = Image.fromarray((world1 * 255).astype('uint8'))

                world2 = world2[0, :].numpy().transpose([1, 2, 0])
                world2 = Image.fromarray((world2 * 255).astype('uint8'))

                if isleft == 1:
                    world1.save(root + "/bevimgs_left/%d.jpg" % int(filename[:-4]))
                elif isleft == 2:
                    world2.save(root + "/bevimgs_right/%d.jpg" % int(filename[:-4]))
                else:
                    final = Image.blend(world1, world2, 0.5)
                    final.save(root + "/bevimgs/%d.jpg" % int(filename[:-4]))



    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'small', 'large'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx', 'robo'])
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--numcam', type=int, default=2)
    parser.add_argument('--aux', type=bool, default=False)
    parser.add_argument('--info', type=str, default="no_extra")
    args = parser.parse_args()

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.ToTensor()])

    data_path = "/home/dzc/Data/MultiviewX"

    data_path = os.path.expanduser(data_path)
    base = Robomaster_1_dataset(data_path, args)
    dataset = oftFrameDataset(base, train=True, transform=train_trans, grid_reduce=1)
    processor = DataProcessor(dataset, data_path)
    # processor.testPrint()
    # processor.cropImage(data_path + "/img", data_path + "/annotations", None, None)
    processor.cropImage(data_path ,data_path + "/Image_subsets/", isleft=3)
    # processor.geneAnno("/home/dzc/Data/1cardata/annotations", "/home/dzc/Data/carONLYdata/annotations")