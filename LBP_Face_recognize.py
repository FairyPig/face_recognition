import numpy as np
import os, math, os.path, glob, random, cv2

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
            hist = cv2.calcHist([np.array(Img, np.uint8)], [0], mask, [256], [0, 256])
            Histogram[:, (i + 1) * (j + 1) - 1] = np.mat(hist).flatten().T
    return Histogram.flatten().T

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
    judgeHistogram = judgeHistogram - mean;
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

def runLBP():
    #加载图像
    FaceMat,type,judgeImgs,judgeType = loadImageSet("G:\FaceRoc\ORL_database\\train\\","G:\FaceRoc\ORL_database\\test\\")
    FaceMat = np.transpose(FaceMat)

    LBPoperator = LBP(FaceMat) #获得实验图像的LBP算子

    #获得实验图像的直方图分布
    exHistograms = np.mat(np.zeros((256*4*7,np.shape(LBPoperator)[1])))

    for i in range(np.shape(LBPoperator)[1]):
        exHistogram = calHistogram(LBPoperator[:,i],7,4)
        exHistograms[:,i] = exHistogram

    exHistograms = exHistograms.transpose()
    mean = np.mean(exHistograms, axis=0)
    exHistograms = exHistograms - mean;
    P = np.dot(exHistograms, exHistograms.transpose())  # 计算P
    v, d = np.linalg.eig(P)  # 计算P的特征值和特征向量
    d = np.dot(exHistograms.transpose(), d)  # 计算Sigma的特征向量 256*4*7 * 320
    train = np.dot(d.transpose(), exHistograms.transpose())  # 计算训练集的主成分值 320*320
    train = train.transpose()
    count = 0
    sum = 0

    for i in range(np.shape(judgeImgs)[0]):
        sum += 1
        it_type = judgeFace(train,mean,d,judgeImgs[i],LBPoperator,7,4)
        if type[it_type] == judgeType[i]:
            count += 1
        else:
            print("图像编号：",i)
            print("本身类型：",judgeType[i])
            print("错分类型：",type[it_type])

    acc = float(count)/float(sum)
    print("人脸分类的正确率：%f"%acc)
if __name__ == '__main__':
    runLBP()