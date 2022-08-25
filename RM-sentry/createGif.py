import imageio
import os
def png2gif(filelist, name, path):
    # filelist为图片列表，存放图片路径
    # name为输出GIF动图的名称 
    # duration表示切换间隔，默认0.5，可以根据需要调整
    frames = []
    for img in filelist:
        frames.append(imageio.imread(os.path.join(path, img)))
    imageio.mimsave(name,frames,'GIF',duration=0.08)


if __name__ == "__main__":
    path = "/home/dzc/Projects/NeuronsDataset/visual_result/sentry/"
    filelist = os.listdir(path)
    filelist.sort(key = lambda x:int(x[:-4]))

    print(filelist[:15])
    png2gif(filelist, "/home/dzc/Desktop/test.gif", path)