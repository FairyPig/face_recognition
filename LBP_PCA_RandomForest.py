# coding=utf-8
import numpy as np
import os, math, os.path, glob, random, cv2
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import time

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

#加载图像
def loadImageSet(add,testpath):
    FaceMat = np.mat(np.zeros((40*8,24*28,)))
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

#算法主过程
def LBP(FaceMat, R = 2, P = 8):
    Region8_x = [-1,0,1,1,1,0,-1,-1]
    Region8_y = [-1,-1,-1,0,1,1,1,0]
    pi = math.pi
    LBPoperator = np.mat(np.zeros(np.shape(FaceMat)))
    for i in range(np.shape(FaceMat)[1]):
        # 对每个图像进行处理
        face = FaceMat[:,i].reshape(28,24)
        W,H = np.shape(face)
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
                    if face[xp,yp]>pixel:
                        repixel += '1'
                    else:
                        repixel += '0'
                # minBinary保持LBP算子旋转不变
                tempface[x, y] = int(minBinary(repixel), base=2)
        LBPoperator[:,i] = tempface.flatten().T
        # cv2.imwrite(str(i)+'hh.jpg',array(tempface,uint8))
    return LBPoperator

# 统计直方图
def calHistogram(ImgLBPope,x_num,y_num):
    Img = ImgLBPope.reshape(28,24)
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
'''
# judgeImg:未知判断图像
# LBPoperator:实验图像LBP算子
# exHistograms:实验图像的直方图分布
def judgeFace_lbp(judgeImg, LBPoperator, exHistograms,x_num,y_num):
    judgeImg = judgeImg.T
    ImgLBPope = LBP(judgeImg)
    #  把图片分为7*4份 , calHistogram返回的直方图矩阵有28个小矩阵内的直方图
    judgeHistogram  = calHistogram(ImgLBPope,x_num,y_num)
    minIndex = 0
    minVals = np.inf

    for i in range(np.shape(LBPoperator)[1]):
        exHistogram = exHistograms[:,i]
        diff = (np.array(exHistogram-judgeHistogram)**2).sum()
        if diff<minVals:
            minIndex = i
            minVals = diff
    return minIndex

# judgeImg:未知判断图像
# LBPoperator:实验图像LBP算子
# exHistograms:实验图像的直方图分布

def judgeFace(train,mean,d,judgeImg, LBPoperator,x_num,y_num):
    judgeImg = judgeImg.T
    ImgLBPope = LBP(judgeImg)
    #  把图片分为7*4份 , calHistogram返回的直方图矩阵有28个小矩阵内的直方图
    judgeHistogram  = calHistogram(ImgLBPope,x_num,y_num)
    judgeHistogram = judgeHistogram - mean.transpose();
    judgeHistogram = np.dot(judgeHistogram.transpose(),d)

    minIndex = 0
    minVals = np.inf

    for i in range(np.shape(LBPoperator)[1]):
        exHistogram = train[i,:]
        diff = (np.array(exHistogram-judgeHistogram)**2).sum()
        if diff<minVals:
            minIndex = i
            minVals = diff
    return minIndex
'''
def runLBP():
    #加载图像
    FaceMat,type,judgeImgs,judgeType = loadImageSet("./dataset/ORL_database/train/","./dataset/ORL_database/test/")
    type = [int(i) for i in type]
    judgeType = [int(j) for j in judgeType]
    FaceMat = np.transpose(FaceMat)#一列一张图
    judgeImgs = np.transpose(judgeImgs)

    LBPoperator = LBP(FaceMat) #获得实验图像的LBP算子 一列是一张图
    LBPoperatortest = LBP(judgeImgs)

    #获得实验图像的直方图分布
    exHistograms = np.mat(np.zeros((256*4*7,np.shape(LBPoperator)[1])))#256*4*7行，320列
    for i in range(np.shape(LBPoperator)[1]):
        exHistogram = calHistogram(LBPoperator[:,i],7,4)
        exHistograms[:,i] = exHistogram
    exHistograms = exHistograms.transpose()

    exHistogramstest = np.mat(np.zeros((256*4*7,np.shape(LBPoperatortest)[1])))
    for i in range(np.shape(LBPoperatortest)[1]):
        exHistogramtest = calHistogram(LBPoperatortest[:,i],7,4)
        exHistogramstest[:,i] = exHistogramtest
    exHistogramstest = exHistogramstest.transpose()
    #mean = np.mean(exHistograms, axis=0)
    #exHistograms = exHistograms - mean;

    n_classes = len(np.unique(type))
    target_names = []
    for i in range(80):
        names = "person" + str(i)
        target_names.append(names)

    m_acc = []
    m_precision = []
    m_recall = []
    m_x = []
    for num in range(1, 200):
        #m_x.append(n_components)
        m_x.append(num)
        n_components = 150
        print("Extracting the top %d eigenfaces from %d faces" % (n_components, exHistograms.shape[0]))
        pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(exHistograms)
        eigenfaces = pca.components_.reshape((n_components, 28, 256))
        X_train_pca = pca.transform(exHistograms)
        X_test_pca = pca.transform(exHistogramstest)
        y_train = []
        for i in range(40):
            for j in range(8):
                y_train.append(i)
        y_test = []
        for i in range(40):
            for j in range(2):
                y_test.append(i)

        acc_array = []
        x = []
        #for num in range(150, 151):

        clf = RandomForestClassifier(n_estimators=num)
        clf.fit(X_train_pca, y_train)
        clf_probs = clf.predict_proba(X_test_pca)
        score = log_loss(y_test, clf_probs)
        result = clf.predict(X_test_pca)

        wrong = 0
        for i in range(80):
            if (result[i] != y_test[i]):
                wrong += 1
                print("错分编号：%d" % i)
                print("原先类别：%d" % y_test[i])
                print("被错分为：%d" % result[i])
                print('\n')
        acc = 1 - wrong.__float__() / 80.0
        acc_array.append(acc)
        m_acc.append(acc)
        m_precision.append(precision_score(y_test, result, average='macro'))
        m_recall.append(recall_score(y_test, result, average='weighted'))
        print("正确率 ：%f" % acc)
        #print("score : %f" % score)

    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(m_x, m_acc)
    # axarr[0].set_xlabel('PCA dimension')
    axarr[0].set_ylabel('accuracy')

    axarr[1].plot(m_x, m_precision)
    # axarr[1].xlabel('PCA dimension')
    axarr[1].set_ylabel('precision')

    axarr[2].plot(m_x, m_recall)
    axarr[2].set_xlabel('tree number')
    axarr[2].set_ylabel('recall')
    plt.show()

if __name__ == '__main__':
    start = time.clock()
    runLBP()
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))