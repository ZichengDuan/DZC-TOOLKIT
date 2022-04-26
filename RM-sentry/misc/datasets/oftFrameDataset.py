import math
import os
import json
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
import torch
from torchvision.transforms import ToTensor
from misc.datasets.Robomaster_1 import *
import warnings
import cv2
from matplotlib import pyplot as plt
from sklearn import preprocessing
warnings.filterwarnings("ignore")

class oftFrameDataset(VisionDataset):
    def __init__(self, base,  train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=1, img_reduce=1, train_ratio=0.9, force_download=True):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 10 / grid_reduce, 10
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce
        self.base = base
        self.train = train
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))
        #
        if train:
            # frame_range = range(0, int(self.num_frame * train_ratio))
            frame_range = range(int(self.num_frame * (1 - train_ratio)), self.num_frame)
        else:
            frame_range = range(0, int(self.num_frame * (1 - train_ratio)))


        # 补充训练
        if train:
            # frame_range = range(0, int(self.num_frame * train_ratio))
            frame_range = range(0, 550)
        else:
            frame_range = range(270, 280)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.score_gt = {}
        self.off_gt = {}
        self.dir_gt = {}
        self.ang_gt = {}
        self.mask = {}
        #
        # self.download_RM_1(frame_range)
        # self.prepare_score_map(frame_range)
        # self.prepare_mask_and_dir_and_offset(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        # if not os.path.exists(self.gt_fpath) or force_download:
        #     self.prepare_gt()

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)

    # 将annotations_positions文件里的json文件里的gt position ID转换成gt(x, y)坐标，再保存一下方向信息，并保存在数据集目录下的gt.txt里
    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                all_pedestrians = [json.load(json_file)]
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                # transfer from ground truth ID into gt (x,y) coordinates
                # grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                grid_x, grid_y= [single_pedestrian['wx'], single_pedestrian['wy']]
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    # 用来生成mask，真值点是1
    def prepare_mask_and_dir_and_offset(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])  # 0, 1, 2, ...
            # 如果在frame的列表里，train和test是分开的
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    all_vehicle = json.load(json_file)
                calculated_pts = []
                mask = np.zeros((int(self.worldgrid_shape[0] / self.grid_reduce), int(self.worldgrid_shape[1] / self.grid_reduce)))
                dir = np.zeros((int(self.worldgrid_shape[0] / self.grid_reduce), int(self.worldgrid_shape[1] / self.grid_reduce)))
                # angle_offsets = np.zeros((int(self.worldgrid_shape[0] / self.grid_reduce), int(self.worldgrid_shape[1] / self.grid_reduce)))
                offsets = np.zeros((2, int(self.worldgrid_shape[0] / self.grid_reduce), int(self.worldgrid_shape[1] / self.grid_reduce)))
                angle_sin_cos = np.zeros((2, int(self.worldgrid_shape[0] / self.grid_reduce), int(self.worldgrid_shape[1] / self.grid_reduce)))
                # seg_img = np.zeros((int(self.worldgrid_shape[0] / self.grid_reduce),int(self.worldgrid_shape[1] / self.grid_reduce), 3), dtype=float)

                for k, single_vehicle in enumerate(all_vehicle):
                    direc = single_vehicle['direc'] # 这辆车的朝向分类
                    ang = single_vehicle['angle'] # 这辆车的精确角度
                    x, y = [single_vehicle['wx'], single_vehicle['wy']]
                    # mm -> cm
                    x, y = float(x / 10), float(y / 10)

                    # ------------------------------------------------------------------
                    x1_ori, x2_ori, x3_ori, x4_ori = x + 25, x + 25, x - 25, x - 25
                    y1_ori, y2_ori, y3_ori, y4_ori = y + 25, y - 25, y - 25, y + 25

                    x1_rot, x2_rot, x3_rot, x4_rot = \
                        int(math.cos(ang) * (x1_ori - x) - math.sin(ang) * (y1_ori - y) + x), \
                        int(math.cos(ang) * (x2_ori - x) - math.sin(ang) * (y2_ori - y) + x), \
                        int(math.cos(ang) * (x3_ori - x) - math.sin(ang) * (y3_ori - y) + x), \
                        int(math.cos(ang) * (x4_ori - x) - math.sin(ang) * (y4_ori - y) + x)

                    y1_rot, y2_rot, y3_rot, y4_rot = \
                        int(math.sin(ang) * (x1_ori - x) + math.cos(ang) * (y1_ori - y) + y), \
                        int(math.sin(ang) * (x2_ori - x) + math.cos(ang) * (y2_ori - y) + y), \
                        int(math.sin(ang) * (x3_ori - x) + math.cos(ang) * (y3_ori - y) + y), \
                        int(math.sin(ang) * (x4_ori - x) + math.cos(ang) * (y4_ori - y) + y)

                    # ------------------------------------------------------------------
                    cords = np.array([[[int(x1_rot / self.grid_reduce), int(y3_rot / self.grid_reduce)],
                                       [int(x2_rot / self.grid_reduce), int(y4_rot / self.grid_reduce)],
                                       [int(x3_rot / self.grid_reduce), int(y1_rot / self.grid_reduce)],
                                       [int(x4_rot / self.grid_reduce), int(y2_rot / self.grid_reduce)]]],
                                     dtype=np.int32)

                    midLine = []
                    for sec in range(8):
                        midLine.append(np.pi / 180 * (22.5 + 45 * sec))
                    # angle_offsets[pt_x, pt_y] = (ang - midLine[direc]) / (np.pi / 8)

                    cv2.fillPoly(mask, cords, 1)
                    cv2.fillPoly(dir, cords, direc + 1)
                    cv2.fillPoly(angle_sin_cos[0], cords, np.sin(ang))
                    cv2.fillPoly(angle_sin_cos[1], cords, np.cos(ang))
                    # seg_img = cv2.fillPoly(seg_img, cords, (255, 255, 255))
                    #-------------------------------------------------------------------
                    points = np.where(mask != 0)
                    for i, (pt_x, pt_y) in enumerate(zip(points[0], points[1])):
                        if (pt_x, pt_y) in calculated_pts:
                            continue
                        else:
                            # 特别注意！wx,wy指的是边长，而不是矩阵里的坐标。所以刚好应该反过来，以和np.where对应,而且因为xy的坐标原点是左下角，所以应该上下颠倒
                            off_x = y / self.grid_reduce - pt_x # 中心点缩放八倍之后的偏差
                            off_y = x / self.grid_reduce - pt_y
                            offsets[0, pt_x, pt_y] = off_x / (25 / self.grid_reduce)
                            offsets[1, pt_x, pt_y] = off_y / (25 / self.grid_reduce)
                # if frame < 28:
                #     print(frame)
                #     np.savetxt("mask%d.csv" % frame, mask, delimiter=",")
                #     print("Saved")
                self.mask[frame] = mask
                self.dir_gt[frame] = dir
                self.off_gt[frame] = offsets
                self.ang_gt[frame] = angle_sin_cos



    # gaussian score map,依据战车中心点生成
    def prepare_score_map(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])  # 0, 1, 2, ...
            # 如果在frame的列表里，train和test是分开的
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    all_vehicle = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                # 对每个行人都遍历
                for single_vehicle in all_vehicle:
                    # 通过id获得世界角度的二维坐标，只有这一个，从上往下看的坐标是固定的。
                    x, y = [single_vehicle['wx'], single_vehicle['wy']]
                    # mm -> cm
                    x = float(x / 10)
                    y = float(y / 10)
                    # 按网格大小和像素的比例缩放，如果是xy的方式那就先x后y，最后i_s, j_s里面存着世界坐标下的gt人的脚的坐标
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    # 在世界地图上用direction标出有位置的地方，之后用coo_matrix来生成这个矩阵
                    v_s.append(1)
                score_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape,dtype=int)
                self.score_gt[frame] = score_map

    # mask范围内的每个点都有一个gt角度（弧度制）
    def prepare_ang_map(self, frame_range):
        pass

    def __getitem__(self, index):
        frame = list(self.score_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)


        score_gt = self.score_gt[frame].toarray()
        off_gt = self.off_gt[frame]
        dir_gt = self.dir_gt[frame]
        ang_gt = self.ang_gt[frame]
        mask = self.mask[frame]

        # np.savetxt("sin.csv", ang_gt[0], delimiter=",")
        # np.savetxt("cos.csv", ang_gt[1], delimiter=",")

        if self.target_transform is not None:
            score_gt = self.target_transform(score_gt)
            a, b, c = off_gt.shape
            off_gt = self.target_transform(off_gt).reshape(a, b, c)
            dir_gt = self.target_transform(dir_gt)
            ang_gt = torch.tensor(ang_gt).reshape(a, b, c)
            mask = self.target_transform(mask)
        # np.savetxt("sin.csv", ang_gt.squeeze()[0].numpy(), delimiter=",")
        # np.savetxt("cos.csv", ang_gt.squeeze()[1].numpy(), delimiter=",")
        return imgs, score_gt.to(torch.float), off_gt.to(torch.float), dir_gt.to(torch.float), ang_gt.to(torch.float), mask.to(torch.float), frame

    def __len__(self):
        return len(self.score_gt.keys())



if __name__ == '__main__':
    test()