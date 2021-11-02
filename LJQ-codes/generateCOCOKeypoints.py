import cv2
import numpy as np
import json
import math
import os
from _ctypes import PyObj_FromPtr
import re
class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr
class KeyPointGenerator():
    def __init__(self, frame_range = range(0, 2805)):
        self.keypoints = {}
        self.frame_range = frame_range
        self.root = "/home/dzc/Data/zed/"
        self.generate3Dkeypoints()


    def keypoint(self, center, rot):
        x, y = center
        # 右为正方向
        # y = Const.grid_height - y
        x1_ori, x2_ori, x3_ori, x4_ori, x6_ori = x+30, x-30, x-30, x+30, x+20
        y1_ori, y2_ori, y3_ori, y4_ori, y6_ori = y+25, y+25, y-25, y-25, y
        # x1_rot, x2_rot, x3_rot, x4_rot, x6_rot = \
        #     int(math.cos(rot) * (x1_ori - x) - math.sin(rot) *
        #             y1_ori - (Const.grid_height - y) + x), \
        #     int(math.cos(rot) * (x2_ori - x) - math.sin(rot) *
        #             y2_ori - (Const.grid_height - y) + x), \
        #     int(math.cos(rot) * (x3_ori - x) - math.sin(rot) *
        #             y3_ori - (Const.grid_height - y) + x), \
        #     int(math.cos(rot) * (x4_ori - x) - math.sin(rot) *
        #             y4_ori - (Const.grid_height - y) + x), \
        #     int(math.cos(rot) * (x6_ori - x) - math.sin(rot) *
        #             y6_ori - (Const.grid_height - y) + x)
        #
        # y1_rot, y2_rot, y3_rot, y4_rot, y6_rot= \
        #     int(math.sin(rot) * (x1_ori - x) + math.cos(rot) * (y1_ori - (Const.grid_height - y)) + (
        #             Const.grid_height - y)), \
        #     int(math.sin(rot) * (x2_ori - x) + math.cos(rot) * (y2_ori - (Const.grid_height - y)) + (
        #             Const.grid_height - y)), \
        #     int(math.sin(rot) * (x3_ori - x) + math.cos(rot) * (y3_ori - (Const.grid_height - y)) + (
        #             Const.grid_height - y)), \
        #     int(math.sin(rot) * (x4_ori - x) + math.cos(rot) * (y4_ori - (Const.grid_height - y)) + (
        #             Const.grid_height - y)), \
        #     int(math.sin(rot) * (x6_ori - x) + math.cos(rot) * (y6_ori - (Const.grid_height - y)) + (
        #             Const.grid_height - y))


        x1_rot, x2_rot, x3_rot, x4_rot, x6_rot = \
            int(math.cos(rot) * (x1_ori - x) - math.sin(rot) *
                    (y1_ori - y) + x), \
            int(math.cos(rot) * (x2_ori - x) - math.sin(rot) *
                    (y2_ori - y) + x), \
            int(math.cos(rot) * (x3_ori - x) - math.sin(rot) *
                    (y3_ori - y) + x), \
            int(math.cos(rot) * (x4_ori - x) - math.sin(rot) *
                    (y4_ori - y) + x), \
            int(math.cos(rot) * (x6_ori - x) - math.sin(rot) *
                    (y6_ori - y) + x)

        y1_rot, y2_rot, y3_rot, y4_rot, y6_rot= \
            int(math.sin(rot) * (x1_ori - x) + math.cos(rot) * (y1_ori - y) + y), \
            int(math.sin(rot) * (x2_ori - x) + math.cos(rot) * (y2_ori - y) + y), \
            int(math.sin(rot) * (x3_ori - x) + math.cos(rot) * (y3_ori - y) + y), \
            int(math.sin(rot) * (x4_ori - x) + math.cos(rot) * (y4_ori - y) + y), \
            int(math.sin(-rot) * (x6_ori - x) + math.cos(-rot) * (y6_ori - y) + y)


        p1 = [x1_rot, y3_rot, 10]
        p2 = [x2_rot, y4_rot, 10]
        p3 = [x3_rot, y1_rot, 10]
        p4 = [x4_rot, y2_rot, 10]
        p5 = [x, y, 35]
        p6 = [x6_rot, y6_rot, 35]

        p1_ori = [x1_ori, y1_ori, 10]
        p2_ori = [x2_ori, y2_ori, 10]
        p3_ori = [x3_ori, y3_ori, 10]
        p4_ori = [x4_ori, y4_ori, 10]
        p5_ori = [x, y, 37]
        p6_ori = [x6_ori, y6_ori, 37]

        # print([p1, p2, p3, p4, p5, p6])
        return [p1, p2, p3, p4, p5, p6]
        # return [p1_ori, p2_ori, p3_ori, p4_ori, p5_ori, p6_ori]

    def generate3Dkeypoints(self):
        cur = 2535
        for fname in sorted(os.listdir("/home/dzc/Data/zed/annotations/")):
            frame = int(fname.split('.')[0])
            frame_keypoints = []
            with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                cars = [json.load(json_file)][0]
            # if frame != cur:
            #     print(fname)
            #     img = cv2.imread("/home/dzc/Data/zed/bevimgs/%d.jpg" % frame)
            for i, car in enumerate(cars):
                wx = int(car["wx"]) // 10
                wy = int(car["wy"]) // 10
                bev_angle = float(car["angle"])
                if frame == cur:
                    print(frame, wx, wy)
                keypoints = self.keypoint([wx, wy], bev_angle)
                frame_keypoints.append(keypoints)
                # if frame == cur:
                #     print(keypoints)
                    # cv2.circle(img, (wx, wy), radius=1, color = (255, 255, 0), thickness=2)
                    # for pts in keypoints:
                    #     x, y, _ = pts
                    #     cv2.circle(img, (x, y), radius=1, color = (255, 255, 0), thickness=2)
                    # cv2.imwrite("/home/dzc/Desktop/Desktop/CASIA/proj/voxelpose-pytorch/data/MVM3D/script/tmp_bevimgs/%d.jpg" % frame, img)
            self.keypoints[str(frame)] = frame_keypoints

    def generateAnnotationJson(self):
        for i in self.frame_range:
            annotation = open("/home/dzc/Desktop/Desktop/CASIA/proj/voxelpose-pytorch/data/MVM3D/data/MVM3D/zed_coco19/%d.json" % (i), 'w')
            data = {}
            bodies = []
            cars = self.keypoints[str(i)]
            if i == 2535:
                print(cars)
            for car_id, car in enumerate(cars):
                per_body_data = {}
                # per_body_data = json.loads(json.dumps(per_body_data))
                per_body_data["id"] = car_id
                per_body_data["joints19"] = list(np.concatenate((np.array(car).reshape(6, -1), np.ones(shape=(6, 1))), axis=1).reshape(-1))
                # print(car)
                if i == 2535:
                    print(per_body_data)
                bodies.append(per_body_data)
            data["bodies"] = bodies
            annotation.write(json.dumps(data, indent=4, cls=MyEncoder))
            annotation.close()


        pass

    def generateCameraJson(self):
        left_intrinsic_matrix, left_extrinsic_matrix, left_coef_matrix = self.get_intrinsic_extrinsic_matrix("left")
        right_intrinsic_matrix, right_extrinsic_matrix, right_coef_matrix = self.get_intrinsic_extrinsic_matrix("right")

        calibration = open("/home/dzc/Desktop/Desktop/CASIA/proj/voxelpose-pytorch/data/MVM3D/data/MVM3D/calibration_ZED.json", 'w')
        data = {}
        data["calibDataSource"] = "ZED"
        left_camera = {}
        left_camera["name"] = "00_00"
        left_camera["type"] = "daheng"
        left_camera["resolution"] = [1280, 720]
        left_camera["K"] = left_intrinsic_matrix.tolist()
        left_camera["distCoef"] = left_coef_matrix.tolist()
        left_camera["R"] = left_extrinsic_matrix[:, :3].tolist()
        left_camera["t"] = left_extrinsic_matrix[:, -1].tolist()
        left_camera["panel"] = 0
        left_camera["node"] = 0

        right_camera = {}
        right_camera["name"] = "00_01"
        right_camera["type"] = "daheng"
        right_camera["resolution"] = [1280, 720]
        right_camera["K"] = right_intrinsic_matrix.tolist()
        right_camera["distCoef"] = right_coef_matrix.tolist()
        right_camera["R"] = right_extrinsic_matrix[:, :3].tolist()
        right_camera["t"] = right_extrinsic_matrix[:, -1].tolist()
        right_camera["panel"] = 0
        right_camera["node"] = 1

        cameras = [left_camera, right_camera]
        data["cameras"] = cameras

        calibration.write(json.dumps(data, indent=4, cls=MyEncoder, ensure_ascii=False))
        calibration.close()

    def get_intrinsic_extrinsic_matrix(self, cam):
        intrinsic_camera_path = os.path.join("/home/dzc/Data/zed/calibration/intrinsic")
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             "intri_"+cam+".xml"),
                                                             flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('intri_matrix').mat()
        intrinsic_params_file.release()

        coef_camera_path = os.path.join("/home/dzc/Data/zed/calibration/coef")
        coef_params_file = cv2.FileStorage(os.path.join(coef_camera_path,
                                                             "coef_"+cam+".xml"),
                                                             flags=cv2.FILE_STORAGE_READ)
        coef_matrix = coef_params_file.getNode('coef_matrix').mat()
        coef_params_file.release()

        extrinsic_camera_path = os.path.join("/home/dzc/Data/zed/calibration/extrinsic")
        extrinsic_params_file = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                             "extri_"+cam+".xml"),
                                                             flags=cv2.FILE_STORAGE_READ)
        extrinsic_matrix = extrinsic_params_file.getNode('extri_matrix').mat()
        extrinsic_params_file.release()
        # print(intrinsic_matrix, extrinsic_matrix)
        for i in range(3):
            extrinsic_matrix[i,3] /= 10
        return intrinsic_matrix, extrinsic_matrix, coef_matrix.reshape(5,)


if __name__ == "__main__":
    generator = KeyPointGenerator(range(0,2805))
    generator.generateAnnotationJson()
    # generator.generateCameraJson()