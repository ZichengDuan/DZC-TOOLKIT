import os
import shutil
# 复制root path中的文件到target path并且根据target path中已有的文件数量对新来的文件重命名
target_path = "/home/dzc/Data/opensource/seperate_anno"
root_path = "/home/dzc/Data/opensource/annotations"
target_file_num = len(os.listdir(target_path))
root_file_num = len(os.listdir(root_path))

print(target_file_num)

i = 0
name = 0
while i < root_file_num:
    idx = str(name)
    # if 6 - len(idx) > 0:
    #     for j in range(6 - len(idx)):
    #         idx = "0" + idx
    filename = os.path.join(root_path, "%s.json" % idx)

    if os.path.exists(filename):
        shutil.copy(filename, os.path.join(target_path, "%d.json" % (i + target_file_num)))
        i += 1
        name += 1
    else:
        name += 1
