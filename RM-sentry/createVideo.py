import cv2
fps = 20
# size = (640, 480)
# size = (1280, 720)
size = (800, 449)
video = cv2.VideoWriter("/home/dzc/Videos/seperate_bev.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)   #视频保存在当前目录下, 格式为 motion-jpeg codec，图片颜色失真比较小

for i in list(range(0, 8660, 10)):
    print(i)
    img = cv2.imread("/home/dzc/Data/opensource/seperate_bev/%d.jpg" % (i))
    video.write(img)

video.release()
print('Video has been made.')