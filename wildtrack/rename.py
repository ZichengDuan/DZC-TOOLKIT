import os
import cv2
for j in range(1, 8):
    path = "/home/dzc/Data/Wildtrack/Image_subsets/C%d" % j
    try:
        list = sorted(os.listdir(path))
        num = 0
        for i in range(400):
            img = cv2.imread(os.path.join(path, list[i]))
            try:
                cur_num = len(os.listdir("/home/dzc/Data/Wildtrack/renamed_images/"))
            except:
                cur_num = 0
            cv2.imwrite("/home/dzc/Data/Wildtrack/renamed_images/" + "%d.png" % (cur_num), img)
    except:
        print("dzc")