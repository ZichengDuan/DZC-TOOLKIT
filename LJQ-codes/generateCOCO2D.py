import json
import os
import numpy as np


class COCO_generator():
    def __init__(self):
        self.bboxes = {}
        self.root = "/home/dzc/Data/opensource/"
        self.read_info()

        pass

    def read_info(self):
        # 读文件存起来，分别读Coco19和MVM3D annotation
        for idx, fname in enumerate(sorted(os.listdir("/home/dzc/Data/opensource/seperate_anno"))):
            frame = int(fname.split('.')[0])
            frame_box = []
            with open(os.path.join(self.root, 'seperate_anno', fname)) as json_file:
                cars = [json.load(json_file)][0]

                for i, car in enumerate(cars):
                    # ymin = int(car["views"][0]["ymin"])
                    # xmin = int(car["views"][0]["xmin"])
                    # ymax = int(car["views"][0]["ymax"])
                    # xmax = int(car["views"][0]["xmax"])
                    if frame < 4330 and int(car["views"][0]["ymin"]) != -1:
                        ymin = int(car["ymin_od"])
                        xmin = int(car["xmin_od"])
                        ymax = int(car["ymax_od"])
                        xmax = int(car["xmax_od"])
                        frame_box.append([ymin, xmin, ymax, xmax])
                    elif frame >= 4330 and int(car["views"][1]["ymin"]) != -1:
                        ymin = int(car["ymin_od"])
                        xmin = int(car["xmin_od"])
                        ymax = int(car["ymax_od"])
                        xmax = int(car["xmax_od"])
                        frame_box.append([ymin, xmin, ymax, xmax])
                self.bboxes[str(idx)] = frame_box

    def get_coco2D(self):
        # 准备封装保存
        target_id = 0
        coco_annotation = open(
            "/home/dzc/Desktop/Desktop/CASIA/proj/voxelpose-pytorch/data/MVM3D/data/MVM3D/coco_bev_perspective_binary.json",
            'w')
        data = {}
        data["info"] = "info"
        data["licenses"] = "licenses"
        images = []
        for idx, file in enumerate(sorted(os.listdir("/home/dzc/Data/opensource/seperate_bev"))):
            cur_img = {}
            img_path = "/home/dzc/Data/opensource/seperate_bev/" + file
            cur_img["license"] = "None"
            cur_img["file_name"] = img_path
            cur_img["coco_url"] = ""
            cur_img["height"] = 449
            cur_img["width"] = 800
            cur_img["date_captured"] = "2021-10-27 11:11:11"
            cur_img["flickr_url"] = ""
            cur_img["id"] = idx
            images.append(cur_img)
        data["images"] = images

        annotation = []
        category = []
        for idx, file in enumerate(sorted(os.listdir("/home/dzc/Data/opensource/seperate_anno"))):
            frame_bbox = self.bboxes[str(idx)]
            for num, bbox in enumerate(frame_bbox):
                if bbox[0] == -1:
                    continue
                cur_anno = {}
                cur_cate = {}
                ymin, xmin, ymax, xmax = bbox
                x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin

                cur_anno["image_id"] = idx
                cur_anno["category_id"] = 0
                cur_anno["segmentation"] = [[]]
                cur_anno["area"] = w * h
                cur_anno["bbox"] = [x, y, w, h]
                cur_anno["id"] = target_id
                cur_anno["is_crowd"] = 0
                cur_anno["category_id"] = 1
                target_id += 1
                annotation.append(cur_anno)

        cur_cate["supercategory"] = "person"
        cur_cate["id"] = 1
        cur_cate["name"] = "person"
        category.append(cur_cate)

        data["annotations"] = annotation
        data["categories"] = category

        coco_annotation.write(json.dumps(data))
        coco_annotation.close()
        print("Generation Complete")


if __name__ == "__main__":
    cocoGenerator = COCO_generator()
    cocoGenerator.get_coco2D()