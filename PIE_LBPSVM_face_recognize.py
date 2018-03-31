# coding=utf-8
import numpy as np
import os, math, os.path, glob, random, cv2
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scipy.io as sio
from skimage.feature import local_binary_pattern
import time
from scipy.stats import itemfreq

# 为了让LBP具有旋转不变性，将二进制串进行旋转。
# 假设一开始得到的LBP特征为10010000，那么将这个二进制特征，
# 按照顺时针方向旋转，可以转化为00001001的形式，这样得到的LBP值是最小的。
# 无论图像怎么旋转，对点提取的二进制特征的最小值是不变的，
# 用最小值作为提取的LBP特征，这样LBP就是旋转不变的了。
def minBinary(pixel):
    length = len(pixel)
    zero = ''
    # range(length)[::-1] 使得i从01234变为43210
    for i in range(length)[::-1]:
        if pixel[i] == '0':
            pixel = pixel[:i]
            zero += '0'
        else:
            return zero + pixel
    if len(pixel) == 0:
        return '0'
'''
#加载图像
def loadImageSet(add,testpath):
    FaceMat = np.mat(np.zeros((40*8,24*28)))
    FaceTestMat = np.mat(np.zeros((40*2,24*28)))
    type = []
    testType = []
    j = 0
    for i in os.listdir(add):
        try:
            img = cv2.imread(add + i)
        except:
            print("load %s failed"%i)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        nimg = np.array(img)
        column = nimg[:,-1]
        nimg = np.column_stack((nimg,column))
        FaceMat[j,:] = np.mat(nimg).flatten()
        j += 1
        type.append(i[1:-6])
    j = 0
    for i in os.listdir(testpath):
        try:
            img = cv2.imread(testpath + i)
        except:
            print("load %s failed"%i)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        nimg = np.array(img)
        column = nimg[:,-1]
        nimg = np.column_stack((nimg,column))
        FaceTestMat[j,:] = np.mat(nimg).flatten()
        j += 1
        testType.append(i[1:-6])
    return FaceMat,type,FaceTestMat,testType
'''
#算法主过程
def LBP(FaceMat, R = 2, P = 8):
    Region8_x = [-1,0,1,1,1,0,-1,-1]
    Region8_y = [-1,-1,-1,0,1,1,1,0]
    pi = math.pi
    LBPoperator = np.mat(np.zeros(np.shape(FaceMat)))
    for i in range(np.shape(FaceMat)[1]):
        # 对每个图像进行处理
        face = FaceMat[:,i].reshape(64,64)
        W, H = np.shape(face)
        tempface = np.mat(np.zeros((W,H)))
        for x in range(R,W-R):
            for y in range(R,H-R):
                repixel = ''
                pixel = int(face[x,y])
                #圆形LBP算子
                for p in [2,1,0,7,6,5,4,3]:
                    p = float(p)
                    xp = x + R * np.cos(2*pi*(p/P))
                    yp = y - R * np.sin(2*pi*(p/P))
                    xp = int(xp)
                    yp = int(yp)
                    if face[xp, yp] > pixel:
                        repixel += '1'
                    else:
                        repixel += '0'
                # minBinary保持LBP算子旋转不变
                tempface[x, y] = int(minBinary(repixel), base=2)
        LBPoperator[:,i] = tempface.flatten().T
        # cv2.imwrite(str(i)+'hh.jpg',array(tempface,uint8))
    return LBPoperator

def LBPcv2(FaceMat, R = 2, P = 8):
    radius = 3
    n_points = 8 * radius
    LBPoperator = np.mat(np.zeros(np.shape(FaceMat)))
    for i in range(np.shape(FaceMat)[1]):
        # 对每个图像进行处理
        face = FaceMat[:,i].reshape(64, 64)
        H, W = np.shape(face)
        #face1 = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(face, n_points, radius)
        lbp1 = np.mat(lbp).flatten().T
        LBPoperator[:, i] = lbp1
    return LBPoperator

# 统计直方图
def calHistogram(ImgLBPope,x_num,y_num):
    Img = ImgLBPope.reshape(64,64)
    W,H = np.shape(Img)
    #把图像分为7*4份
    Histogram = np.mat(np.zeros((256,x_num*y_num)))
    maskx,masky = W/x_num,H/y_num
    for i in range(x_num):
        for j in range(y_num):
            # 使用掩膜opencv来获得子矩阵直方图
            mask = np.zeros(np.shape(Img), np.uint8)
            mask[int(i * maskx): int((i + 1) * maskx), int(j * masky):int((j + 1) * masky)] = 255
            hist = cv2.calcHist([np.array(Img, np.uint8)], [0], mask, [256], [0, 255])
            Histogram[:, i * y_num + j] = np.mat(hist).flatten().T      #矩阵28行，256列，28个图像块，每个图像块256维直方图特征
    return Histogram.flatten().T    #返回一列

def runLBP():
    mat_file05 = 'G:\FaceRoc\PIE dataset\PIE dataset\Pose05_64x64'
    mat_file07 = 'G:\FaceRoc\PIE dataset\PIE dataset\Pose07_64x64'
    mat_file09 = 'G:\FaceRoc\PIE dataset\PIE dataset\Pose09_64x64'
    mat_file27 = 'G:\FaceRoc\PIE dataset\PIE dataset\Pose27_64x64'
    mat_file29 = 'G:\FaceRoc\PIE dataset\PIE dataset\Pose29_64x64'

    data_pose05 = sio.loadmat(mat_file05)
    data_pose07 = sio.loadmat(mat_file07)
    data_pose09 = sio.loadmat(mat_file09)
    data_pose27 = sio.loadmat(mat_file27)
    data_pose29 = sio.loadmat(mat_file29)
    train_num = np.sum(data_pose05['isTest'] == 0)
    test_num = np.sum(data_pose05['isTest'] == 1)

    train_num += np.sum(data_pose07['isTest'] == 0)
    test_num += np.sum(data_pose07['isTest'] == 1)

    train_num += np.sum(data_pose09['isTest'] == 0)
    test_num += np.sum(data_pose09['isTest'] == 1)

    train_num += np.sum(data_pose27['isTest'] == 0)
    test_num += np.sum(data_pose27['isTest'] == 1)

    train_num += np.sum(data_pose29['isTest'] == 0)
    test_num += np.sum(data_pose29['isTest'] == 1)

    FaceMat = np.mat(np.zeros((train_num, 64 * 64)))
    judgeImgs = np.mat(np.zeros((test_num, 64 * 64)))

    m = 0
    n = 0
    for i in range(np.shape(data_pose05['isTest'])[0]):
        if data_pose05['isTest'][i] == 0:
            FaceMat[m] = data_pose05['fea'][i]
            m = m + 1
        else:
            judgeImgs[n] = data_pose05['fea'][i]
            n = n + 1

    for i in range(np.shape(data_pose07['isTest'])[0]):
        if data_pose07['isTest'][i] == 0:
            FaceMat[m] = data_pose07['fea'][i]
            m = m + 1
        else:
            judgeImgs[n] = data_pose07['fea'][i]
            n = n + 1

    for i in range(np.shape(data_pose09['isTest'])[0]):
        if data_pose09['isTest'][i] == 0:
            FaceMat[m] = data_pose09['fea'][i]
            m = m + 1
        else:
            judgeImgs[n] = data_pose09['fea'][i]
            n = n + 1

    for i in range(np.shape(data_pose27['isTest'])[0]):
        if data_pose27['isTest'][i] == 0:
            FaceMat[m] = data_pose27['fea'][i]
            m = m + 1
        else:
            judgeImgs[n] = data_pose27['fea'][i]
            n = n + 1

    for i in range(np.shape(data_pose29['isTest'])[0]):
        if data_pose29['isTest'][i] == 0:
            FaceMat[m] = data_pose29['fea'][i]
            m = m + 1
        else:
            judgeImgs[n] = data_pose29['fea'][i]
            n = n + 1
    type = [0 for x in range(train_num)]
    judgeType = [0 for x in range(test_num)]
    m = 0
    n = 0
    for i in range(np.shape(data_pose05['isTest'])[0]):
        if data_pose05['isTest'][i] == 0:
            type[m] = data_pose05['gnd'][i]
            m = m + 1
        else:
            judgeType[n] = data_pose05['gnd'][i]
            n = n + 1

    for i in range(np.shape(data_pose07['isTest'])[0]):
        if data_pose07['isTest'][i] == 0:
            type[m] = data_pose07['gnd'][i]
            m = m + 1
        else:
            judgeType[n] = data_pose07['gnd'][i]
            n = n + 1

    for i in range(np.shape(data_pose09['isTest'])[0]):
        if data_pose09['isTest'][i] == 0:
            type[m] = data_pose09['gnd'][i]
            m = m + 1
        else:
            judgeType[n] = data_pose09['gnd'][i]
            n = n + 1

    for i in range(np.shape(data_pose27['isTest'])[0]):
        if data_pose27['isTest'][i] == 0:
            type[m] = data_pose27['gnd'][i]
            m = m + 1
        else:
            judgeType[n] = data_pose27['gnd'][i]
            n = n + 1

    for i in range(np.shape(data_pose29['isTest'])[0]):
        if data_pose29['isTest'][i] == 0:
            type[m] = data_pose29['gnd'][i]
            m = m + 1
        else:
            judgeType[n] = data_pose29['gnd'][i]
            n = n + 1

    type = [int(i) for i in type]
    judgeType = [int(j) for j in judgeType]
    # 加载图像
    # FaceMat,type,judgeImgs,judgeType = loadImageSet("E:/face\ORL_database\\train\\","E:/face\ORL_database\\test\\")
    # type = [int(i) for i in type]
    # judgeType = [int(j) for j in judgeType]
    FaceMat = np.transpose(FaceMat)  # 一列一张图
    judgeImgs = np.transpose(judgeImgs)
    FaceMat.astype(int)
    judgeImgs.astype(int)
    LBPoperator = LBPcv2(FaceMat)  # 获得实验图像的LBP算子 一列是一张图
    LBPoperatortest = LBPcv2(judgeImgs)

    # 获得实验图像的直方图分布
    exHistograms = np.mat(np.zeros((256 * 2 * 2, np.shape(LBPoperator)[1])))  # 256*4*7行，320列
    for i in range(np.shape(LBPoperator)[1]):
        exHistogram = calHistogram(LBPoperator[:, i], 2, 2)
        exHistograms[:, i] = exHistogram
    exHistograms = exHistograms.transpose()

    exHistogramstest = np.mat(np.zeros((256 * 2 * 2, np.shape(LBPoperatortest)[1])))
    for i in range(np.shape(LBPoperatortest)[1]):
        exHistogramtest = calHistogram(LBPoperatortest[:, i], 2, 2)
        exHistogramstest[:, i] = exHistogramtest
    exHistogramstest = exHistogramstest.transpose()
    # mean = np.mean(exHistograms, axis=0)
    # exHistograms = exHistograms - mean;

    n_classes = len(np.unique(type))
    target_names = []
    for i in range(len(np.unique(type))):
        names = "person" + str(i)
        target_names.append(names)

    n_components = 150
    print("Extracting the top %d eigenfaces from %d faces" % (n_components, exHistograms.shape[0]))
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(exHistograms)
    eigenfaces = pca.components_.reshape((n_components, 4, 256))
    X_train_pca = pca.transform(exHistograms)
    X_test_pca = pca.transform(exHistogramstest)
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, type)
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    y_pred = clf.predict(X_test_pca)
    print(classification_report(judgeType, y_pred, target_names=target_names))
    print(confusion_matrix(judgeType, y_pred, labels=range(n_classes)))

    for i in range(test_num):
        if judgeType[i] != y_pred[i]:
            print("错分编号：%d" % i)
            print("原先类别：%d" % judgeType[i])
            print("被错分为：%d" % y_pred[i])

if __name__ == '__main__':
    start = time.clock()
    runLBP()
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))