import numpy as np
import cv2

def trans_intri_to_xml(lpath,rpath):
    l_file = cv2.FileStorage("/home/dzc/Data/zed/calibration/intri_left.xml", cv2.FILE_STORAGE_WRITE)
    l = open(lpath, 'r')
    lmat = l.readline().split( )
    lmat = [float(val) for val in lmat]
    lmat = np.array(lmat).reshape((3,3))
    l_file.write("intri_matrix", lmat)
    l_file.release()

    # r_file = cv2.FileStorage("/home/dzc/Data/mix/calibration2/intri_right.xml", cv2.FILE_STORAGE_WRITE)
    # r = open(rpath, 'r')
    # rmat = r.readline().split()
    # rmat = [float(val) for val in rmat]
    # rmat = np.array(rmat).reshape((3, 3))
    # r_file.write("intri_matrix", rmat)
    # r_file.release()

def trans_extri_to_xml(lpath, rpath):
    l_file = cv2.FileStorage("/home/dzc/Data/zed/calibration/extri_left.xml", cv2.FILE_STORAGE_WRITE)
    l = open(lpath, 'r')
    # print(l.read().split('\n'))
    l_rot_mat,l_trans_mat = [mat.split( ) for mat in l.read().split('\n')]
    # print(l_rot_mat,l_trans_mat)
    for i, val in enumerate(l_rot_mat):
        l_rot_mat[i] = float(val)

    for i, val in enumerate(l_trans_mat):
        l_trans_mat[i] = float(val)

    l_rot_mat = np.array(l_rot_mat).reshape((3,3))
    l_trans_mat = np.array(l_trans_mat).reshape((3,1))

    l_extri = np.hstack((l_rot_mat,l_trans_mat))

    l_file.write("extri_matrix", l_extri)
    l_file.release()


    # r_file = cv2.FileStorage("/home/dzc/Data/mix/calibration2/extri_right.xml", cv2.FILE_STORAGE_WRITE)
    # r = open(rpath, 'r')
    # r_rot_mat,r_trans_mat = [mat.split( ) for mat in r.read().split('\n')[:-1]]
    # for i, val in enumerate(r_rot_mat):
    #     r_rot_mat[i] = float(val)
    #
    # for i, val in enumerate(r_trans_mat):
    #     r_trans_mat[i] = float(val)
    #
    # r_rot_mat = np.array(r_rot_mat).reshape((3,3))
    # r_trans_mat = np.array(r_trans_mat).reshape((3,1))
    #
    # r_extri = np.hstack((r_rot_mat,r_trans_mat))
    # r_file.write("extri_matrix", r_extri)
    # r_file.release()


    print("finish")


if __name__ == "__main__":
    trans_intri_to_xml("/home/dzc/Data/zed/left-in.txt", "/home/dzc/Data/zed/right-in.txt")
    trans_extri_to_xml("/home/dzc/Data/zed/left-ex.txt", "/home/dzc/Data/zed/right-ex.txt")