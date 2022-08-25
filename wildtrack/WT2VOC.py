import json, re, os
import numpy as np
import cv2
import xml.etree.ElementTree as ET

intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']

def project(proj, pts):
    pts_hom = np.insert(pts, 3, 1)
    pts = proj @ pts_hom
    pts /= pts[-1]
    return pts[:2]

class Wt2VOC():
    def __init__(self):
        self.root = "/home/dzc/Data/Wildtrack"
        self.num_cam = 7
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])
        print("dzc")

    def read_boxes(self):
        all_boxes = {
            "cam1": [[] for i in range(400)],
            "cam2": [[] for i in range(400)],
            "cam3": [[] for i in range(400)],
            "cam4": [[] for i in range(400)],
            "cam5": [[] for i in range(400)],
            "cam6": [[] for i in range(400)],
            "cam7": [[] for i in range(400)],
            }
        
        for frame_id, fname in enumerate(sorted(os.listdir(os.path.join(self.root, 'annotations_positions')))):
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
                for single_pedestrian in all_pedestrians:
                    for view_id, view in enumerate(single_pedestrian['views']):
                        if view['xmax'] == view['ymax'] == view['xmin'] == view['ymin'] == -1:
                            continue

                        feet_coord = self.get_worldcoord_from_pos(single_pedestrian['positionID'])
                        feet_coord = np.insert(feet_coord, 2, 0)
                        feet_pts = project(proj = self.intrinsic_matrices[view['viewNum']] @ self.extrinsic_matrices[view['viewNum']], pts = feet_coord)

                        # save to a seperate file
                        all_boxes["cam%d" % (view_id + 1)][frame_id].append({'xmin': view['xmin'], 
                                                                    'ymin': view['ymin'],
                                                                    'xmax': view['xmax'],
                                                                    'ymax': view['ymax'],
                                                                    'standing_point': feet_pts})

        return all_boxes
    
    def split_annotation(self, boxes):
        for i in range(7):
            cur_view_per_frame_box = boxes["cam%d" % (i + 1)]
            for m in range(400):
                try:
                    cur_num = len(os.listdir("/home/dzc/Data/Wildtrack/per_image_annotation/"))
                except:
                    cur_num = 0
                print(cur_num + 1)
                annotation = open("/home/dzc/Data/Wildtrack/per_image_annotation/%d.json" % (cur_num), 'w')
                frame_data = []
                
                # print(cur_view_per_frame_box[0])
                for id, box in enumerate(cur_view_per_frame_box[0]):
                    per_pedestrian_data = {}
                    per_pedestrian_data = json.loads(json.dumps(per_pedestrian_data))
                    per_pedestrian_data["Pedestrian_id"] = id

                    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                    bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
                    per_pedestrian_data["bbox"] = bbox

                    per_pedestrian_data["standing_point"] = {"x": box["standing_point"][0], "y": box["standing_point"][1]}
                    frame_data.append(per_pedestrian_data)

                annotation.write(json.dumps(frame_data, indent=4))
                annotation.close()

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
    # boxes = testvoc.read_boxes()
    # testvoc.split_annotation(boxes)
    testvoc.visualize(0)
    