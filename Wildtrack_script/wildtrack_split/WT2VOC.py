import json, re, os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image

intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']

region = [0, 0, 480, 1440]

def project(proj, pts):
    pts_hom = np.insert(pts, 3, 1)
    pts = proj @ pts_hom
    pts /= pts[-1]
    return pts[:2]

class Wt2VOC():
    def __init__(self):
        self.root = "D:\\2.study\\Data\\Wildtrack"
        self.num_cam = 7
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])
        print("dzc")

    def read_boxes(self):
        if os.path.exists("all_boxes.json"):
            all_boxes = open("all_boxes.json", "r")
            all_boxes = json.load(all_boxes)
            return all_boxes

        all_boxes = {
            "cam1": [[] for i in range(400)],
            "cam2": [[] for i in range(400)],
            "cam3": [[] for i in range(400)],
            "cam4": [[] for i in range(400)],
            "cam5": [[] for i in range(400)],
            "cam6": [[] for i in range(400)],
            "cam7": [[] for i in range(400)],
            }
        xi = np.arange(0, 480, 1)
        yi = np.arange(0, 1440, 1)
        world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
        world_coord = self.get_worldcoord_from_worldgrid(world_grid)
        edges = [None for i in range(7)]
        for frame_id, fname in enumerate(sorted(os.listdir(os.path.join(self.root, 'annotations_positions')))):
            print(frame_id)
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
                for single_pedestrian in all_pedestrians:
                    for view_id, view in enumerate(single_pedestrian['views']):
                        if view['xmax'] == view['ymax'] == view['xmin'] == view['ymin'] == -1:
                            continue
                        feet_coord = self.get_worldcoord_from_pos(single_pedestrian['positionID'])
                        feet_coord = np.insert(feet_coord, 2, 0)
                        feet_pts = project(proj = self.intrinsic_matrices[view['viewNum']] @ self.extrinsic_matrices[view['viewNum']], pts = feet_coord)
                        
                        if edges[view_id] is not None:
                            edge = edges[view_id]
                        else:
                            # calculate region in image
                            img_coord = self.get_imagecoord_from_worldcoord(world_coord, self.intrinsic_matrices[view['viewNum']],
                                                                                    self.extrinsic_matrices[view['viewNum']])
                            img_coord = img_coord[:, np.where((img_coord[0] >= 0) & (img_coord[1] >= 0) &
                                                                (img_coord[0] < 1920) & (img_coord[1] < 1080))[0]]
                            img_coord = img_coord.astype(int).transpose()

                            # edges
                            # left most edges
                            left_most_pts = img_coord[img_coord[:, 0] < min(img_coord[:, 0]) + 5]
                            left_most_max, left_most_min = left_most_pts[left_most_pts[:, 1] == max(left_most_pts[:, 1])][0].reshape(1, -1), left_most_pts[left_most_pts[:, 1] == min(left_most_pts[:, 1])][0].reshape(1, -1)

                            # right most edges
                            right_most_pts = img_coord[img_coord[:, 0] > max(img_coord[:, 0]) - 5]
                            right_most_max, right_most_min = right_most_pts[right_most_pts[:, 1] == max(right_most_pts[:, 1])][0].reshape(1, -1), right_most_pts[right_most_pts[:, 1] == min(right_most_pts[:, 1])][0].reshape(1, -1)

                            # top most edges
                            top_most_pts = img_coord[img_coord[:, 1] < min(img_coord[:, 1]) + 5]
                            top_most_max, top_most_min = top_most_pts[top_most_pts[:, 0] == max(top_most_pts[:, 0])][0].reshape(1, -1), top_most_pts[top_most_pts[:, 0] == min(top_most_pts[:, 0])][0].reshape(1, -1)
                                
                            # bottom most edges
                            bt_most_pts = img_coord[img_coord[:, 1] > max(img_coord[:, 1]) - 5]
                            bt_most_max, bt_most_min = bt_most_pts[bt_most_pts[:, 0] == max(bt_most_pts[:, 0])][0].reshape(1, -1), bt_most_pts[bt_most_pts[:, 0] == min(bt_most_pts[:, 0])][0].reshape(1, -1)

                            edge = np.concatenate((left_most_min, top_most_min, top_most_max, right_most_min, right_most_max, bt_most_max, bt_most_min, left_most_max), axis=0)

                            edges[view_id] = edge
                        # save to a seperate file
                        all_boxes["cam%d" % (view_id + 1)][frame_id].append({'xmin': view['xmin'], 
                                                                    'ymin': view['ymin'],
                                                                    'xmax': view['xmax'],
                                                                    'ymax': view['ymax'],
                                                                    'standing_point': feet_pts.tolist(),
                                                                    'region': edge.tolist()})
        
        tf = open("all_boxes.json", "w")
        json.dump(all_boxes,tf)
        tf.close()
        
        return all_boxes
    
    def split_annotation(self, boxes):
        for i in range(7):
            cur_view_per_frame_box = boxes["cam%d" % (i + 1)]
            for m in range(400):
                try:
                    cur_num = len(os.listdir("D:\\2.study\\Data\\Wildtrack\\per_image_annotation"))
                except:
                    cur_num = 0
                print(cur_num + 1)
                with open("D:\\2.study\\Data\\Wildtrack\\per_image_annotation\\%d.json" % (cur_num), 'w') as annotation:
                    frame_data = {}
                    # print(cur_view_per_frame_box[0])
                    frame_data["version"] = "4.5.6"
                    frame_data["flags"] = {}
                    shapes = []
                    for id, box in enumerate(cur_view_per_frame_box[m]):
                        per_pedestrian_data = {}
                        per_pedestrian_data["label"] = 1
                        per_pedestrian_data["points"] = [[box["xmin"], box["ymin"]],[box["xmax"], box["ymax"]]]
                        per_pedestrian_data["shape_type"] = "rectangle"
                        per_pedestrian_data["flags"] = {}
                        per_pedestrian_data["text"] = str(id)

                        per_kpt_data ={}
                        per_kpt_data["label"] = 2
                        per_kpt_data["points"] = [[box["standing_point"][0], box["standing_point"][1]]]
                        per_kpt_data["shape_type"] = "point"
                        per_kpt_data["flags"] = {}
                        per_kpt_data["text"] = str(id)
                        shapes.append(per_pedestrian_data)
                        shapes.append(per_kpt_data)

                    # the region polygon
                    edges = box['region']
                    shapes.append({"label": 3, "points": edges, "group_id": "null", "shape_type": "polygon", "flags": {}})
                    frame_data["shapes"] = shapes
                    frame_data["imagePath"] = f'{cur_num}.png'
                    frame_data["imageHeight"] = 1080
                    frame_data["imageWidth"] = 1920
                    json.dump(frame_data, annotation, indent=4)


    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    @staticmethod
    def get_worldgrid_from_pos(pos):
        grid_x = pos % 480
        grid_y = pos // 480
        return np.array([grid_x, grid_y], dtype=int)

    @staticmethod
    def get_worldcoord_from_worldgrid(worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        dim = worldgrid.shape[0]
        if dim == 2:
            grid_x, grid_y = worldgrid
            coord_x = -300 + 2.5 * grid_x
            coord_y = -900 + 2.5 * grid_y
            return np.array([coord_x, coord_y])
        elif dim == 3:
            grid_x, grid_y, grid_z = worldgrid
            coord_x = -300 + 2.5 * grid_x
            coord_y = -900 + 2.5 * grid_y
            coord_z = 2.5 * grid_z
            return np.array([coord_x, coord_y, coord_z])

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix

    def get_imagecoord_from_worldcoord(self, world_coord, intrinsic_mat, extrinsic_mat):
        project_mat = intrinsic_mat @ extrinsic_mat
        project_mat = np.delete(project_mat, 2, 1)
        world_coord = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
        image_coord = project_mat @ world_coord
        image_coord = image_coord[:2, :] / image_coord[2, :]
        return image_coord

    def visualize(self, idx):
        img = cv2.imread(os.path.join(self.root, 'renamed_images', "%d.png" % idx))
        with open(os.path.join(self.root, 'per_image_annotation', "%d.json" % idx)) as json_file:
            pedestrians = json.load(json_file)
            for single_pedestrain in pedestrians:
                box = single_pedestrain["bbox"]
                xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

                st_pt = single_pedestrain["standing_point"]

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
                print(int(st_pt["x"]), int(st_pt["y"]))
                cv2.circle(img, (int(st_pt["x"]), int(st_pt["y"])), radius=10, thickness=-1, color=(0, 0, 255))
        cv2.imwrite("test_visual.jpg", img)


if __name__ == "__main__":
    testvoc = Wt2VOC()
    boxes = testvoc.read_boxes()
    testvoc.split_annotation(boxes)
    # testvoc.visualize(0)
    