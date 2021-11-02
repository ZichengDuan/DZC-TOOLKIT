import cv2
import os
def renamejpg(path, out):
    for i in range(0, 770):
        print(i)
        idx = str(i)
        if 4 - len(idx) > 0:
            for j in range(4 - len(idx)):
                idx = "0" + idx
        # img = cv2.txt(path + "/frame%s.jpg" % idx)
        # cv2.imwrite(out + "/frame%s.jpg" % str(i + 1934), img)
        os.rename(path + "/frame%s.jpg" % idx, out + "/frame%s.jpg" % str(1934 + i))
# renamejpg("/home/dzc/Data/ZCdata/img/imgright1")
renamejpg("/home/dzc/Downloads/new_raw/img/left", "/home/dzc/Data/ZCdata/img/left1")
renamejpg("/home/dzc/Downloads/new_raw/img/right", "/home/dzc/Data/ZCdata/img/right2")
